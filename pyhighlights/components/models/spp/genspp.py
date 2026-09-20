from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import chain
from multiprocessing import TimeoutError as MPTimeoutError
from multiprocessing import get_all_start_methods, get_context
from multiprocessing.pool import ThreadPool
from typing import Any, Dict, Iterable, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp.base import SPP, SPPBackbone, SPPSelector
from pyhighlights.utility import diagnostics


class GenSPP(SPP):
    """Single-generator SPP evaluated through external genetic search.

    The generator is not trained by gradient descent: a genetic search over
    generator parameters scores each candidate by training a fresh predictor
    on it, which removes the cooperative equilibrium the other SPP models
    have to fight. See :class:`GenSPPTrainer` for the search itself.

    Ruggeri and Signorelli, 2025, *Interlocking-free Selective Rationalization
    Through Genetic-based Learning*, ACL 2025, <https://aclanthology.org/2025.acl-long.59/>.
    Reference implementation: <https://github.com/nlp-unibo/gen-spp>.
    """

    def __init__(self, **kwargs):
        selector_backbones = kwargs.get("selector_backbones")
        predictor_backbone = kwargs.get("predictor_backbone")
        same_backbone_config = selector_backbones == predictor_backbone
        super().__init__(**kwargs)
        if len(self.selectors) != 1:
            raise ValueError("GenSPP requires exactly one selector")
        if len(self.losses) != 1 or self.losses[0].name != "classification":
            raise ValueError("GenSPP requires one classification task loss")
        generator = {
            id(parameter)
            for parameter in chain(
                self.selector_backbones.parameters(), self.selectors.parameters()
            )
        }
        predictor = {
            id(parameter)
            for parameter in chain(
                self.predictor_backbone.parameters(), self.predictor.parameters()
            )
        }
        if generator & predictor:
            raise ValueError(
                "GenSPP requires independent generator and predictor parameters"
            )
        if same_backbone_config:
            source = [
                *(
                    parameter
                    for parameter in self.selector_backbone.parameters()
                    if not parameter.requires_grad
                ),
                *self.selector_backbone.buffers(),
            ]
            target = [
                *(
                    parameter
                    for parameter in self.predictor_backbone.parameters()
                    if not parameter.requires_grad
                ),
                *self.predictor_backbone.buffers(),
            ]
            if len(source) != len(target) or any(
                left.shape != right.shape for left, right in zip(source, target)
            ):
                raise ValueError("GenSPP backbones have incompatible frozen state")
            with th.no_grad():
                for left, right in zip(source, target):
                    right.copy_(left)

    def select(
        self,
        data: InputData,
        selector: SPPSelector,
        backbone: SPPBackbone,
    ) -> tuple[th.Tensor, th.Tensor]:
        # The encoder reads its own subtokens and the selection is made over
        # words, exactly as `SPP.select` does it. What this override leaves out
        # is the empty-selection fallback: a search scores an empty selection
        # rather than repairing it.
        states = self.to_words(
            backbone.encode(data.features, self.encoder_mask(data)), data
        )
        highlight_logits = selector(states)
        highlight_mask = highlight_logits.argmax(dim=-1).to(highlight_logits.dtype)
        valid = self.selection_valid(data).to(highlight_mask.dtype)
        return highlight_logits, highlight_mask * valid

    def generator_parameters(self) -> list[th.nn.Parameter]:
        return [
            parameter
            for parameter in chain(
                self.selector_backbones.parameters(), self.selectors.parameters()
            )
            if parameter.requires_grad
        ]

    def predictor_parameters(self) -> list[th.nn.Parameter]:
        return [
            parameter
            for parameter in chain(
                self.predictor_backbone.parameters(), self.predictor.parameters()
            )
            if parameter.requires_grad
        ]

    def on_train_epoch_start(self) -> None:
        # Training mode is set on the whole model. The generator is frozen
        # while a predictor is fitted on it, and dropout inside it would score
        # the same candidate differently from one epoch to the next.
        super().on_train_epoch_start()
        self.selector_backbones.eval()
        self.selectors.eval()
        # And so would dropout inside a predictor backbone that descent never
        # moves -- `GenSPPTransformerBackboneConfig` sets
        # `freeze_transformer`, and a frozen Hugging Face encoder left in
        # training mode still drops. That made a candidate's fitness a
        # property of the random state as well as of its chromosome, which is
        # the one thing a search cannot have: the same chromosome scored twice
        # would not agree with itself.
        if not any(
            parameter.requires_grad
            for parameter in self.predictor_backbone.parameters()
        ):
            self.predictor_backbone.eval()

    def configure_optimizers(self):
        # Gradient descent only ever reaches the predictor: the generator is
        # searched, not trained, so training this model on its own fits a
        # predictor to whatever selection its untrained generator makes.
        return self.build_optimizer([(self.predictor_parameters(), 1.0)])


@dataclass
class _Individual:
    chromosome: th.Tensor
    fitness: float
    task_loss: float
    selection_rate: float


#: What a worker scores against: the trainer, and the two splits
#: :meth:`GenSPPTrainer._fit` froze. Set when the worker starts, from
#: arguments fork gives it by inheritance. Only the tasks themselves cross a
#: pipe, so a generation costs a chromosome each way.
_WORK: Tuple["GenSPPTrainer", Any, Any] | None = None


def _score_in_worker(item: Tuple[int, th.Tensor | None, th.device]):
    """One candidate, in a process of its own. Module level to be picklable."""
    if _WORK is None:  # pragma: no cover -- a worker that never forked
        raise RuntimeError("worker started without a search to score for")
    trainer, train_loader, val_loader = _WORK
    _, chromosome, device = item
    individual, model = trainer._evaluate_individual(
        train_loader, val_loader, chromosome, device
    )
    return individual, trainer._trained_state(model)


#: How long a worker gets to answer the probe below. Generous, because it is
#: paid once per search and a loaded node can be slow to schedule a fork --
#: and short against a search measured in hours, which is what it protects.
#: Forking a process that already has threads is what Python warns about and
#: what can deadlock a child on a lock held elsewhere at the moment of the
#: fork. The probe cannot prevent that; a timeout on it turns the hang into a
#: search that runs on threads instead.
PROBE_SECONDS = 60.0


def _worker_can_train() -> bool:
    """Whether a worker can train at all.

    Torch refuses the combination once autograd has run threads in the parent,
    and it refuses it in the **child**, when the pass is attempted, rather than
    at the fork. So a search asks a worker to train something trivial before
    trusting it with a candidate.

    The probe descends as well as differentiates. An optimizer step reaches
    torch's accelerator health check, which asks the current accelerator for
    its stream and so initialises CUDA even for parameters that live on the
    CPU. In a forked child that raises, and a probe that only ran backward
    would pass and leave the first real candidate to fail.
    """
    parameter = th.nn.Parameter(th.zeros(1))
    optimizer = th.optim.Adam([parameter])
    (parameter * 2).sum().backward()
    optimizer.step()
    return True


def _start_worker(trainer: "GenSPPTrainer", train_loader: Any, val_loader: Any) -> None:
    """Take delivery of the search, once, and keep torch to one thread.

    Eight workers each taking a thread per core is sixty-four threads over
    however many the machine has, which costs more in contention than the
    threads can return on a model this size.
    """
    global _WORK
    _WORK = (trainer, train_loader, val_loader)
    th.set_num_threads(1)


class GenSPPTrainer:
    """External search matching released GenSPP's selection-rate objective."""

    def __init__(
        self,
        model: RegistrationKey[GenSPP],
        n_generations: int = 100,
        population_size: int = 50,
        selection_rate: float = 0.5,
        mutation_probability: float = 1.0,
        mutation_std: float = 0.05,
        threshold_mutation_std: float | None = None,
        predictor_epochs: int = 3,
        task_loss_limit: float = 0.1,
        stop_threshold: float = 0.01,
        seed: int | None = None,
        devices: Sequence[str] = ("cpu",),
    ):
        if n_generations < 0:
            raise ValueError("n_generations must be non-negative")
        if population_size < 2 or population_size % 2:
            raise ValueError("population_size must be an even integer of at least two")
        if not 0.0 < selection_rate <= 1.0:
            raise ValueError("selection_rate must be in (0, 1]")
        if int(selection_rate * population_size) < 1:
            raise ValueError(
                f"selection_rate {selection_rate} over a population of "
                f"{population_size} draws no couple, so a generation has no "
                "children"
            )
        if not 0.0 < mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in (0, 1]")
        if not math.isfinite(mutation_std) or mutation_std <= 0:
            raise ValueError("mutation_std must be finite and greater than zero")
        if threshold_mutation_std is not None and (
            not math.isfinite(threshold_mutation_std) or threshold_mutation_std <= 0
        ):
            raise ValueError(
                "threshold_mutation_std must be finite and greater than zero"
            )
        if predictor_epochs < 1:
            raise ValueError("predictor_epochs must be positive")
        if not math.isfinite(task_loss_limit) or task_loss_limit < 0:
            raise ValueError("task_loss_limit must be finite and non-negative")
        if not math.isfinite(stop_threshold) or stop_threshold <= 0:
            raise ValueError("stop_threshold must be finite and greater than zero")

        self.model = model
        self.n_generations = n_generations
        self.population_size = population_size
        #: How many couples a generation draws, as a share of the population.
        #: Each couple crosses into two children, so the release's 0.5 adds one
        #: child per member -- ``int(0.5 * 50) = 25`` couples and 50 children,
        #: which then compete with their 50 parents for 50 places.
        self.selection_rate = selection_rate
        self.mutation_probability = mutation_probability
        self.mutation_std = mutation_std
        #: Standard deviation of the perturbation a mutation applies to the
        #: selector's decision threshold, or ``None`` to perturb it at
        #: ``mutation_std`` like every other gene.
        #:
        #: The threshold is the head's output bias. A head emitting two
        #: logits decides on their difference, so perturbing both at ``s``
        #: gives the threshold a perturbation of ``s * sqrt(2)``; this
        #: parameter names the standard deviation of that difference and the
        #: per-gene value is derived from it. The released GenSPP sets it to
        #: 0.10 against a 0.05 elsewhere, which its paper does not report, so
        #: the default here follows the paper and a reproduction of the
        #: release sets this instead.
        self.threshold_mutation_std = threshold_mutation_std
        self.predictor_epochs = predictor_epochs
        self.task_loss_limit = task_loss_limit
        self.stop_threshold = stop_threshold
        self.seed = seed
        if not devices:
            raise ValueError("devices must name at least one device")
        # One worker per device, which is the same knob for both cases the
        # search is run under: ``("cpu",) * 8`` is eight candidates at once on
        # eight cores, ``("cuda:0", "cuda:1")`` is a node's cards. A candidate
        # is small enough that splitting one across devices would cost more
        # than it saves, so a device runs a whole candidate. Whether a worker
        # is a process or a thread is :meth:`_score`'s decision, not this one's.
        self.devices = [th.device(device) for device in devices]

        self.population: list[_Individual] = []
        #: One entry per generation: the best **objective** reached in it,
        #: which is ``1 / fitness`` and therefore falls as the search
        #: improves. It is what ``stop_threshold`` is compared against. The
        #: name is the released implementation's rather than a description,
        #: and it is serialized under itself inside ``search.json``, so it is
        #: left alone rather than renamed under existing readers.
        self.training_progress: list[float] = []
        self._best_model: GenSPP | None = None
        self._best_fitness = -math.inf
        self._initial_state: dict[str, th.Tensor] | None = None
        self._embeddings: th.Tensor | None = None
        self._random = random.Random()
        self._torch_generator = th.Generator()
        #: The process pool, while a search is running. See :meth:`_open_pool`.
        self._pool = None
        #: Trailing genes carrying the decision threshold, counted off the
        #: first candidate in :meth:`_fit`. Zero until then, so a mutation
        #: outside a search treats every gene alike.
        self._threshold_genes = 0

    @staticmethod
    def compute_fitness(
        task_loss: float,
        selection_rate: float,
        task_loss_limit: float,
    ) -> float:
        """Selection rate traded against task loss, higher being better.

        ``task_loss_limit`` is the cross entropy above which a candidate is not
        competing at all: it gets the floor of 1.0 whatever it selected, so the
        search cannot buy a sparse selection with a model that has stopped
        classifying. The paper sets it per corpus -- 0.1 on the toy corpus,
        which is nearly solved, and 0.6 on HateXplain, which is not.

        Below the limit the objective is
        ``1 - sqrt((1 - selection_rate) * (1 - task_loss))``, and the fitness
        is its reciprocal: both terms have to be small for it to be large, so a
        candidate cannot win on sparsity alone.
        """
        if task_loss > task_loss_limit:
            return 1.0
        objective = 1.0 - math.sqrt(
            (1.0 - selection_rate) * (1.0 - min(task_loss, 1.0))
        )
        return 1.0 / (objective + 1e-8)

    def fit(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
        embeddings: th.Tensor | None = None,
    ) -> GenSPP:
        """Search for a generator, and return the best model the search found.

        ``embeddings`` is the corpus's token table, when the task read one. It
        is passed rather than registered for the reason
        :class:`~pyhighlights.components.tasks.SPPTask` passes it: a matrix is
        data, and no configuration should carry one. Every candidate loads the
        same table, before :meth:`_align_initial_state` sees it, so it is
        shared state rather than part of a chromosome.
        """
        with th.random.fork_rng(devices=self._cuda_indices()):
            return self._fit(train_loader, val_loader, embeddings)

    def _fit(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
        embeddings: th.Tensor | None = None,
    ) -> GenSPP:
        if self.seed is None:
            # Drawn rather than left alone, and installed on the global
            # generator too. Candidates are built from that one, and `fit`
            # restores it around the whole search -- so an unseeded search
            # that only reseeded its own generators drew the same founders
            # every time, with only its selection and mutation differing.
            drawn = th.seed()
            self._random.seed(drawn)
            self._torch_generator.manual_seed(drawn)
        else:
            self._random.seed(self.seed)
            self._torch_generator.manual_seed(self.seed)
            th.manual_seed(self.seed)

        # One batch order for the whole search, drawn after the seeding above.
        # A shuffling `DataLoader` re-permutes on every pass, so a chromosome
        # would score differently depending on how many candidates preceded it.
        # Validation is materialised for cost rather than order: `_evaluate`
        # sums over every batch, and a search re-collates the split thousands
        # of times. Both splits stay in memory.
        train_batches = list(train_loader)
        val_batches = list(val_loader)
        if not train_batches:
            raise ValueError("training loader must contain at least one batch")
        if not val_batches:
            raise ValueError("validation loader must contain at least one batch")

        self._best_model = None
        self._best_fitness = -math.inf
        # Before any pool exists. Left to the first candidate, two workers
        # could both find it unset and each install its own model's state.
        self._initial_state = None
        self._embeddings = embeddings
        self._threshold_genes = self._count_threshold_genes(self._candidate())
        try:
            self._open_pool(train_batches, val_batches)
            return self._search(train_batches, val_batches)
        finally:
            self._close_pool()

    def _search(
        self,
        train_batches: List[InputData],
        val_batches: List[InputData],
    ) -> GenSPP:
        """The search proper, with the workers open around it."""
        # Chromosomes first, drawn here from the search's own random state,
        # and scored after. Scoring cannot draw them: it does not read the
        # global generator at all, by the requirement several devices rest on,
        # so every founder would come back with the model's shared initial
        # state and the first generation would be one point repeated.
        founders = [self._founder_chromosome() for _ in range(self.population_size)]
        self.population = self._score(train_batches, val_batches, founders)
        self.training_progress.clear()

        for generation in range(self.n_generations):
            diagnostics.record("generation", index=generation)
            self._run_generation(train_batches, val_batches)
            best_loss = 1.0 / self._best_individual().fitness
            self.training_progress.append(best_loss)
            if best_loss <= self.stop_threshold:
                break

        if self._best_model is None:
            raise RuntimeError("GenSPP search produced no model")
        self._best_model.to("cpu")
        self._best_model.eval()
        # The winner's generator is a chromosome the search settled on, and
        # nothing moves it again: scoring it is a forward pass, and a second
        # search draws its own founders rather than resuming this one. Saying
        # so on the model is what lets a reader of it -- a cost table counting
        # what gradient descent moves, a caller building an optimizer over
        # `parameters()` -- tell the searched half from the trained one.
        for parameter in self._best_model.generator_parameters():
            parameter.requires_grad_(False)
        return self._best_model

    def _candidate(self) -> GenSPP:
        """One model of the searched key, ready to carry a chromosome.

        The embedding table is loaded before the shared state is aligned,
        because the table is part of that shared state: a candidate built
        after the first would otherwise be asked to load a state dict whose
        embedding has a different number of rows.
        """
        model = Registry.from_key(self.model, expected_type=GenSPP)
        if self._embeddings is not None:
            model.load_embeddings(self._embeddings)
        self._align_initial_state(model)
        return model

    def _with_chromosome(self, chromosome: th.Tensor | None) -> GenSPP:
        """A candidate carrying these genes, or the state it was built with.

        ``None`` is the first candidate of a search, whose own initialisation
        is what :meth:`_align_initial_state` then holds every later one to.
        """
        model = self._candidate()
        parameters = model.generator_parameters()
        if not parameters:
            raise ValueError("GenSPP generator has no evolvable parameters")
        if chromosome is None:
            return model
        if chromosome.numel() != sum(parameter.numel() for parameter in parameters):
            raise ValueError("chromosome size does not match generator parameters")
        offset = 0
        with th.no_grad():
            for parameter in parameters:
                size = parameter.numel()
                parameter.copy_(chromosome[offset : offset + size].view_as(parameter))
                offset += size
        return model

    def _founder_chromosome(self) -> th.Tensor:
        """A generator drawn at random, for a member of the first generation."""
        return self._chromosome(self._candidate()).clone()

    def _score(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
        chromosomes: List[th.Tensor],
    ) -> List[_Individual]:
        """Score candidates, one device each, and record the best of them.

        **Processes on CPU, threads on CUDA.** A candidate is a small model,
        so its cost is the training loop stepping from Python rather than the
        arithmetic inside torch -- and that loop holds the GIL. Threads
        therefore buy about 1.4x on eight workers rather than eight, and a
        search measured on eight cores ran at 240% of a possible 800%. On CUDA
        the picture is the other way round: the kernels do release the GIL, and
        a process per device would pay for a CUDA context each and cannot be
        forked from a parent that has already initialised one.

        A candidate takes its device from its position, so a run naming one
        device is the sequential search and pays for no pool at all.
        """
        work = [
            (index, chromosome, self.devices[index % len(self.devices)])
            for index, chromosome in enumerate(chromosomes)
        ]

        def score(item):
            index, chromosome, device = item
            diagnostics.record("candidate", index=index, device=str(device))
            individual, model = self._evaluate_individual(
                train_loader, val_loader, chromosome, device
            )
            return individual, self._trained_state(model)

        # A diagnosed search scores one candidate at a time whatever it was
        # given: the stages report in the order they run and nothing else says
        # which candidate a line belongs to, so two workers writing at once
        # produce a record of one model that never existed. The search a task
        # agrees to diagnose is a handful of candidates wide.
        if len(self.devices) == 1 or diagnostics.active():
            scored = [score(item) for item in work]
        elif self._pool is not None:
            scored = self._pool.map(_score_in_worker, work)
        else:
            with ThreadPool(processes=len(self.devices)) as pool:
                scored = pool.map(score, work)

        # Sequentially, and after every worker has finished: `_best_fitness` is
        # trainer state, and two workers improving on it at once would lose one
        # of the two.
        individuals = []
        for individual, state in scored:
            if individual.fitness > self._best_fitness:
                self._best_fitness = individual.fitness
                self._best_model = self._restored(individual.chromosome, state)
            individuals.append(individual)
        return individuals

    def _forkable(self) -> bool:
        """Whether this search's candidates can be scored in processes.

        CPU only, because a CUDA context cannot be inherited across a fork,
        and only where the platform offers fork at all: spawning would re-import
        and re-register everything per worker, per generation.

        A parent that has already initialised CUDA rules out fork even when
        every device here is a CPU one. Torch marks such a child as forked from
        a CUDA process and refuses to initialise CUDA in it, and an optimizer
        step initialises CUDA whenever an accelerator is present, whatever the
        parameters sit on. A task that trains on a GPU around the search
        therefore leaves the search on threads.
        """
        return (
            all(device.type == "cpu" for device in self.devices)
            and "fork" in get_all_start_methods()
            and not th.cuda.is_initialized()
        )

    def _open_pool(
        self, train_loader: Iterable[InputData], val_loader: Iterable[InputData]
    ) -> None:
        """One process per device, forked once for the whole search.

        The fork is what makes this cheap: a worker reads the corpus and the
        trainer out of inherited memory rather than over a pipe, so only a
        chromosome goes in and a scored candidate comes back.

        Forked here rather than per generation because torch refuses the
        combination outright once autograd has run threads in the parent --

            RuntimeError: Unable to handle autograd's threading in
            combination with fork-based multiprocessing.

        -- and the parent trains nothing itself, so the only clean moment is
        before the first candidate. A caller that has already trained
        something in this process still gets that error, in the worker and not
        at the fork, which is why a pool is asked to differentiate something
        trivial before it is trusted with a candidate. A pool that cannot is
        closed, and the search falls back to the threads it used to use.

        The probe is also what a deadlock runs into. Forking a process that
        already has threads is unsafe in general -- a child can inherit a lock
        no thread of its own will ever release.

        ``forkserver`` is the start method that exists to avoid exactly that,
        by forking workers from a small server process started clean, and it
        cannot be used here. The registry is process-global state, built once
        by the caller, and a worker that did not inherit it cannot build the
        model a chromosome is for::

            NotExpandedException: The registration graph has yet to be
            expanded! Configuration retrieval is not allowed.

        Rebuilding it per worker would cost seconds each and would register a
        second copy of every configuration, which is the failure this project
        already carries a cinnamon floor for. Inheriting memory is what makes
        these workers correct, not merely cheap.

        So the risk is bounded rather than removed. The timeout is the bound:
        a pool that cannot answer inside :data:`PROBE_SECONDS` -- deadlocked,
        or on a node out of memory or descriptors -- is abandoned for threads,
        rather than hanging a search otherwise measured in hours. What it does
        not reach is a parent that hangs inside ``fork`` itself. Measured
        against that: a search opens its pool with one Python thread running,
        torch's being native, which is why CPython's own warning about forking
        a multi-threaded process does not fire outside a test runner that adds
        threads of its own.
        """
        if len(self.devices) == 1 or diagnostics.active() or not self._forkable():
            return
        try:
            # Named as the workers' inputs rather than left in a global
            # for them to find. Under fork these are inherited and not
            # pickled -- a closure `pickle` refuses arrives intact -- so
            # handing over the corpus costs nothing. What it buys is that a
            # launch which fails leaves nothing here still holding it, and
            # that a worker's inputs are written down rather than being
            # whatever the parent happened to have set.
            self._pool = get_context("fork").Pool(
                processes=len(self.devices),
                initializer=_start_worker,
                initargs=(self, train_loader, val_loader),
            )
            self._pool.apply_async(_worker_can_train).get(timeout=PROBE_SECONDS)
        except (RuntimeError, OSError, TimeoutError, MPTimeoutError):
            self._close_pool()

    def _close_pool(self) -> None:
        # Joined rather than left to the collector: `peak_memory` reads
        # `RUSAGE_CHILDREN`, which stays at zero until a child is reaped, and
        # a run's cost is measured as soon as its search returns.
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    @staticmethod
    def _trained_state(model: GenSPP) -> Dict[str, th.Tensor]:
        """What descent moved in a candidate, small enough to send back.

        Everything the model holds except what is frozen, which is the token
        embedding table: GloVe at 25 dimensions over HateXplain's vocabulary
        is megabytes, it is identical in every candidate, and
        :meth:`_candidate` loads it from :attr:`_embeddings` anyway. The
        generator is in the chromosome, and is kept here as well because it
        costs the same as naming it.
        """
        frozen = {
            id(parameter)
            for parameter in model.parameters()
            if not parameter.requires_grad
        }
        return {
            name: tensor.detach().cpu()
            for name, tensor in model.state_dict(keep_vars=True).items()
            if id(tensor) not in frozen
        }

    def _restored(self, chromosome: th.Tensor, state: Dict[str, th.Tensor]) -> GenSPP:
        """The scored model again, from its chromosome and what descent moved.

        Rebuilt rather than kept even where the model never left this process,
        so the search keeps one winner however its candidates were scored.
        """
        model = self._with_chromosome(chromosome)
        model.load_state_dict(state, strict=False)
        return model

    def _best_individual(self) -> _Individual:
        return max(self.population, key=lambda individual: individual.fitness)

    def _evaluate_individual(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
        chromosome: th.Tensor | None,
        device: th.device,
    ) -> Tuple[_Individual, GenSPP]:
        """Score one candidate on one device.

        **Nothing here may depend on the global random state**, which is what
        lets several candidates run at once: torch's default generator is
        process-wide, so what a thread drew from it would depend on how the
        threads interleaved. Nothing does — the model's own initialisation is
        entirely overwritten, by ``_initial_state`` outside the chromosome and
        by the chromosome inside it, the batches arrive as the list :meth:`_fit`
        froze rather than as a loader that would shuffle them again, and the
        registered configurations set ``dropout_rate`` to 0. A dropout rate
        above zero would take that back silently, so
        ``test_scoring_a_candidate_does_not_read_the_global_random_state``
        asserts it.

        It does **advance** that state, because building a model draws from it.
        That is harmless here and measured rather than assumed: the search's
        own randomness is on explicit generators -- ``self._random`` for
        selection and crossover, ``self._torch_generator`` for mutation and
        survival -- so no decision reads what scoring left behind, and
        :meth:`fit` forks the global state so a caller's is restored.
        """
        model = self._with_chromosome(chromosome)
        self._train_predictor(model, train_loader, device)
        task_loss, selection_rate = self._evaluate(model, val_loader, device)
        fitness = self.compute_fitness(
            task_loss=task_loss,
            selection_rate=selection_rate,
            task_loss_limit=self.task_loss_limit,
        )
        if not math.isfinite(fitness):
            # Here, where the candidate and the device that trained it are
            # still in hand. A non-finite fitness survives into the population
            # and fails a generation later inside `random.choices` with
            # `Total of weights must be finite`, which names neither.
            raise ValueError(
                f"candidate on {device} scored a non-finite fitness: "
                f"task loss {task_loss}, selection rate {selection_rate}"
            )
        individual = _Individual(
            chromosome=self._chromosome(model).clone(),
            fitness=fitness,
            task_loss=task_loss,
            selection_rate=selection_rate,
        )
        # The caller records the best, sequentially. Doing it here would be a
        # race between workers on `_best_fitness`.
        return individual, model

    @staticmethod
    def _cuda_index(device: th.device) -> int:
        return device.index if device.index is not None else th.cuda.current_device()

    def _cuda_indices(self) -> list[int]:
        """The CUDA devices `fork_rng` has to save, which may be none."""
        return [
            self._cuda_index(device) for device in self.devices if device.type == "cuda"
        ]

    def _align_initial_state(self, model: GenSPP) -> None:
        """Give every candidate the predictor the first one started from.

        A candidate's fitness has to be a property of its chromosome. It is
        not, if each candidate trains a predictor drawn fresh from the global
        random state: the same chromosome then scores differently depending on
        how many candidates were evaluated before it, and the search ranks
        initialisations alongside genes.

        The released implementation reaches the same place from the other
        side. It keeps a pool of models and resets each reused one to *that
        slot's* initial weights, so a candidate's predictor depends on which
        slot it was given. One shared state removes the dependency entirely.

        The generator is overwritten by the chromosome immediately after this,
        so what this fixes is the predictor, the frozen tensors and the
        buffers.
        """
        evolving = self._generator_names(model)
        if self._initial_state is None:
            self._initial_state = {
                name: tensor.detach().clone()
                for name, tensor in model.state_dict().items()
                if name not in evolving
            }
            return
        missing, unexpected = model.load_state_dict(self._initial_state, strict=False)
        if unexpected or set(missing) != evolving:
            raise ValueError("GenSPP candidates have incompatible state")

    @staticmethod
    def _generator_names(model: GenSPP) -> set:
        """The state-dict keys the chromosome owns.

        Everything else is shared, so these are the only entries a candidate
        may differ in. By identity rather than by name prefix: which modules
        count as the generator is the model's answer, in
        ``generator_parameters``, and a prefix would be a second one.
        """
        evolving = {id(parameter) for parameter in model.generator_parameters()}
        return {
            name
            for name, parameter in model.named_parameters()
            if id(parameter) in evolving
        }

    def _train_predictor(
        self, model: GenSPP, train_loader: Iterable[InputData], device: th.device
    ) -> None:
        generator_parameters = model.generator_parameters()
        if not model.predictor_parameters():
            raise ValueError("GenSPP predictor has no trainable parameters")

        # Freezing the generator is what makes this a candidate evaluation
        # rather than end-to-end training: the optimizer already leaves those
        # parameters out, and this leaves the backward pass out too.
        for parameter in generator_parameters:
            parameter.requires_grad_(False)
        try:
            model.to(device)
            optimizer = model.configure_optimizers()
            for _ in range(self.predictor_epochs):
                model.train(True)
                # Which puts the frozen generator back in eval mode, so its
                # dropout does not score the same candidate two ways.
                model.on_train_epoch_start()
                for batch in train_loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    loss, _ = model.compute_loss(batch, model.training_forward(batch))
                    loss.backward()
                    optimizer.step()
            model.to("cpu")
        finally:
            for parameter in generator_parameters:
                parameter.requires_grad_(True)

    def _evaluate(
        self, model: GenSPP, val_loader: Iterable[InputData], device: th.device
    ) -> tuple[float, float]:
        model.to(device)
        model.eval()
        total_loss = 0.0
        total_rate = 0.0
        sample_count = 0

        with th.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                output = model(batch)
                batch_size = batch.y_true.shape[0]
                loss, _ = model.compute_loss(batch, output)
                total_loss += loss.item() * batch_size
                # The axis the selection was made on, which is the word axis
                # unless the model was told otherwise. `mask` is the word axis
                # always, so a model selecting over subtokens would count
                # subtokens over a word count and feed the search a rate that
                # is not one.
                valid = model.selection_valid(batch).sum(dim=-1).clamp_min(1)
                rates = output.highlight_mask[:, 0].sum(dim=-1) / valid
                total_rate += rates.sum().item()
                sample_count += batch_size

        model.to("cpu")
        if sample_count == 0:
            raise ValueError("validation loader must contain at least one sample")
        return total_loss / sample_count, total_rate / sample_count

    def _run_generation(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
    ) -> None:
        """Breed one generation and keep ``population_size`` of the result.

        Couples are drawn by roulette wheel -- fitness-proportional, with
        replacement, so a good chromosome can parent several children -- and
        each couple crosses into two. ``selection_rate`` is therefore about
        *couples*: at 0.5 a population of fifty draws twenty-five of them and
        so gains fifty children, which then compete with their fifty parents
        for fifty places.

        Parents are kept in the pool rather than replaced. A generation cannot
        lose ground: the best chromosome so far is still a candidate for
        survival against everything its children became.
        """
        weights = [individual.fitness for individual in self.population]
        children = []
        for _ in range(int(self.selection_rate * self.population_size)):
            parent_1, parent_2 = self._random.choices(
                self.population, weights=weights, k=2
            )
            child_1, child_2 = self._crossover(parent_1.chromosome, parent_2.chromosome)
            children.extend([self._mutate(child_1), self._mutate(child_2)])

        # One pool for the whole generation rather than one per couple: every
        # child of a generation is independent of every other.
        scored = self._score(train_loader, val_loader, children)
        self.population = self._select_survivors([*self.population, *scored])

    def _select_survivors(self, candidates: list[_Individual]) -> list[_Individual]:
        """Half elitism: the best half kept outright, the rest drawn by fitness.

        The top ``population_size // 2`` survive because they are the top; the
        other half is sampled from everything below them, fitness-proportional
        and **without replacement**, so a mediocre chromosome has a chance and
        no chromosome takes two places. That is what keeps a search of a
        hundred generations from collapsing onto one lineage.

        ``population_size`` is even by construction, so the two halves are the
        whole population and no candidate below them is kept.
        """
        candidates = sorted(
            candidates, key=lambda individual: individual.fitness, reverse=True
        )
        half = self.population_size // 2
        elites = candidates[:half]
        remainder = candidates[half:]
        weights = th.tensor(
            [individual.fitness for individual in remainder], dtype=th.float64
        )
        selected = th.multinomial(
            weights,
            num_samples=half,
            replacement=False,
            generator=self._torch_generator,
        ).tolist()
        return [*elites, *(remainder[index] for index in selected)]

    def _crossover(
        self, chromosome_1: th.Tensor, chromosome_2: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor]:
        """One-point crossover, returning both children of the cut.

        The cut is anywhere in the chromosome, which is the generator's
        parameters flattened into one vector -- so it usually falls inside a
        weight matrix rather than between two of them.
        """
        point = self._random.randrange(chromosome_1.numel())
        return (
            th.cat((chromosome_1[:point], chromosome_2[point:])),
            th.cat((chromosome_2[:point], chromosome_1[point:])),
        )

    @staticmethod
    def _count_threshold_genes(model: GenSPP) -> int:
        """How many trailing genes carry the selectors' decision threshold.

        The selectors name those parameters through
        :meth:`~pyhighlights.components.models.spp.base.SPPSelector.threshold_parameters`.
        Counted here are the ones that land at the end of the flattened
        chromosome, because :meth:`_mutate` gives its own deviation to a
        trailing slice of that vector. A selector that keeps its threshold
        anywhere else, or that declares none, contributes nothing and is
        searched with one deviation throughout -- rather than having whatever
        parameter happens to be last mutated in its place.
        """
        declared = {
            id(parameter)
            for selector in model.selectors
            for parameter in selector.threshold_parameters()
        }
        genes = 0
        for parameter in reversed(model.generator_parameters()):
            if id(parameter) not in declared:
                break
            genes += parameter.numel()
        return genes

    def _mutate(self, chromosome: th.Tensor) -> th.Tensor:
        """Gaussian noise on a share of the genes, in place of a resample.

        ``mutation_probability`` is per gene and defaults to 1.0, so every
        gene is perturbed: the release mutates the whole chromosome and relies
        on ``mutation_std`` being small to keep a child near its parents.

        ``threshold_mutation_std`` gives the trailing threshold genes a
        standard deviation of their own. It names the perturbation of the
        threshold rather than of one gene, so it is divided by the square root
        of how many genes carry it.
        """
        selected = (
            th.rand(chromosome.shape, generator=self._torch_generator)
            < self.mutation_probability
        )
        noise = th.randn(chromosome.shape, generator=self._torch_generator)
        deviation = th.full_like(chromosome, self.mutation_std)
        genes = self._threshold_genes
        if self.threshold_mutation_std is not None and genes:
            deviation[-genes:] = self.threshold_mutation_std / math.sqrt(genes)
        return chromosome + selected.to(chromosome.dtype) * noise * deviation

    @staticmethod
    def _chromosome(model: GenSPP) -> th.Tensor:
        """The generator's parameters as one flat vector.

        Only the generator: the predictor is fitted by gradient descent on
        every candidate and is not inherited, which is what makes a fitness a
        property of the chromosome rather than of the descent that followed it.
        """
        return th.nn.utils.parameters_to_vector(model.generator_parameters()).detach()
