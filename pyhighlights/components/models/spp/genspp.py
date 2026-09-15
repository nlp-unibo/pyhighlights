from __future__ import annotations

import math
import random
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from itertools import chain
from multiprocessing.pool import ThreadPool
from typing import Iterable, List, Tuple

import lightning as L
import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp.base import SPP, SPPBackbone, SPPSelector


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
        # Lightning puts the whole model in training mode. The generator is
        # frozen while a predictor is fitted on it, and dropout inside it would
        # score the same candidate differently from one epoch to the next.
        super().on_train_epoch_start()
        self.selector_backbones.eval()
        self.selectors.eval()

    def configure_optimizers(self):
        # Gradient descent only ever reaches the predictor: the generator is
        # searched, not trained, so training this model on its own fits a
        # predictor to whatever selection its untrained generator makes.
        return self.build_optimizer([(self.predictor_parameters(), 1.0)])


class _Batches:
    """One stream of batches, not a collection of dataloaders.

    Lightning reads a sequence of loaders as several to combine, so a plain
    list of batches -- what a caller writes in a test, or when the batches are
    already in memory -- has to say it is a single loader.
    """

    def __init__(self, batches: Sequence[InputData]):
        self.batches = batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


@dataclass
class _Individual:
    chromosome: th.Tensor
    fitness: float
    task_loss: float
    selection_rate: float


class GenSPPTrainer:
    """External search matching released GenSPP's selection-rate objective."""

    def __init__(
        self,
        model: RegistrationKey[GenSPP],
        n_generations: int = 100,
        population_size: int = 50,
        mutation_probability: float = 1.0,
        mutation_std: float = 0.05,
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
        if not 0.0 < mutation_probability <= 1.0:
            raise ValueError("mutation_probability must be in (0, 1]")
        if not math.isfinite(mutation_std) or mutation_std <= 0:
            raise ValueError("mutation_std must be finite and greater than zero")
        if predictor_epochs < 1:
            raise ValueError("predictor_epochs must be positive")
        if not math.isfinite(task_loss_limit) or task_loss_limit < 0:
            raise ValueError("task_loss_limit must be finite and non-negative")
        if not math.isfinite(stop_threshold) or stop_threshold <= 0:
            raise ValueError("stop_threshold must be finite and greater than zero")

        self.model = model
        self.n_generations = n_generations
        self.population_size = population_size
        self.mutation_probability = mutation_probability
        self.mutation_std = mutation_std
        self.predictor_epochs = predictor_epochs
        self.task_loss_limit = task_loss_limit
        self.stop_threshold = stop_threshold
        self.seed = seed
        if not devices:
            raise ValueError("devices must name at least one device")
        # One worker per device, which is the same knob for both cases the
        # search is run under: ``("cpu",) * 8`` is the released implementation's
        # CPU thread pool, ``("cuda:0", "cuda:1")`` is a node's cards. A
        # candidate is small enough that splitting one across devices would
        # cost more than it saves, so a device runs a whole candidate.
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
        self._random = random.Random()
        self._torch_generator = th.Generator()

    @staticmethod
    def compute_fitness(
        task_loss: float,
        selection_rate: float,
        task_loss_limit: float,
    ) -> float:
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
    ) -> GenSPP:
        if isinstance(train_loader, Iterator) or isinstance(val_loader, Iterator):
            raise ValueError("train and validation loaders must be re-iterable")

        with th.random.fork_rng(devices=self._cuda_indices()):
            return self._fit(train_loader, val_loader)

    def _fit(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
    ) -> GenSPP:
        if self.seed is None:
            self._random.seed()
            self._torch_generator.seed()
        else:
            self._random.seed(self.seed)
            self._torch_generator.manual_seed(self.seed)
            th.manual_seed(self.seed)

        # Drawn once, here, and handed to every candidate as a list. A
        # `DataLoader` over a training split shuffles, so re-iterating it draws
        # a new permutation from the global random state: the same chromosome
        # then scores differently depending on how many candidates came before
        # it, which is the defect `_align_initial_state` closes on the other
        # side. Threads make it worse rather than introducing it -- with a pool
        # the permutation a candidate gets depends on how the workers
        # interleaved. One order, fixed before the search starts, is what makes
        # a fitness a property of its chromosome.
        #
        # After the seeding above, so the order is the seed's and a rerun
        # repeats it.
        #
        # The validation batches are held for a different reason: `_evaluate`
        # sums over all of them, so their order never mattered, but a loader
        # re-collates the split once per candidate and a search has thousands
        # of those.
        # ponytail: both splits stay in memory for the search; stream them
        # again if a corpus arrives that does not fit.
        train_batches = list(train_loader)
        val_batches = list(val_loader)
        if not train_batches:
            raise ValueError("training loader must contain at least one batch")

        self._best_model = None
        self._best_fitness = -math.inf
        # Before any pool exists. Left to the first candidate, two workers
        # could both find it unset and each install its own model's state.
        self._initial_state = None
        self._align_initial_state(Registry.from_key(self.model, expected_type=GenSPP))
        # Chromosomes first, drawn here from the search's own random state,
        # and scored after. Scoring cannot draw them: it does not read the
        # global generator at all, by the requirement several devices rest on,
        # so every founder would come back with the model's shared initial
        # state and the first generation would be one point repeated.
        founders = [self._founder_chromosome() for _ in range(self.population_size)]
        self.population = self._score(train_batches, val_batches, founders)
        self.training_progress.clear()

        for _ in range(self.n_generations):
            self._run_generation(train_batches, val_batches)
            best_loss = 1.0 / self._best_individual().fitness
            self.training_progress.append(best_loss)
            if best_loss <= self.stop_threshold:
                break

        if self._best_model is None:
            raise RuntimeError("GenSPP search produced no model")
        self._best_model.to("cpu")
        self._best_model.eval()
        return self._best_model

    def _founder_chromosome(self) -> th.Tensor:
        """A generator drawn at random, for a member of the first generation."""
        model = Registry.from_key(self.model, expected_type=GenSPP)
        self._align_initial_state(model)
        return self._chromosome(model).clone()

    def _score(
        self,
        train_loader: Iterable[InputData],
        val_loader: Iterable[InputData],
        chromosomes: List[th.Tensor],
    ) -> List[_Individual]:
        """Score candidates, one device each, and record the best of them.

        Threads rather than processes, as the released implementation does:
        the work is inside torch, which releases the GIL, and a process would
        have to ship a model back over a pipe. A candidate takes its device
        from its position, so a run naming one device is the sequential search
        and pays for no pool.
        """
        work = [
            (chromosome, self.devices[index % len(self.devices)])
            for index, chromosome in enumerate(chromosomes)
        ]

        def score(item):
            return self._evaluate_individual(train_loader, val_loader, *item)

        if len(self.devices) == 1:
            scored = [score(item) for item in work]
        else:
            with ThreadPool(processes=len(self.devices)) as pool:
                scored = pool.map(score, work)

        # Sequentially, and after every worker has finished: `_best_fitness` is
        # trainer state, and two workers improving on it at once would lose one
        # of the two.
        individuals = []
        for individual, model in scored:
            if individual.fitness > self._best_fitness:
                self._best_fitness = individual.fitness
                self._best_model = model
            individuals.append(individual)
        return individuals

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
        model = Registry.from_key(self.model, expected_type=GenSPP)
        self._align_initial_state(model)
        parameters = model.generator_parameters()
        if not parameters:
            raise ValueError("GenSPP generator has no evolvable parameters")
        if chromosome is not None:
            if chromosome.numel() != sum(parameter.numel() for parameter in parameters):
                raise ValueError("chromosome size does not match generator parameters")
            offset = 0
            with th.no_grad():
                for parameter in parameters:
                    size = parameter.numel()
                    parameter.copy_(
                        chromosome[offset : offset + size].view_as(parameter)
                    )
                    offset += size

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

    def _lightning(self, device: th.device) -> L.Trainer:
        """A throwaway trainer for one candidate.

        The search keeps nothing but the weights it ends up with, so logs,
        checkpoints, progress bars and sanity checks are all off: a run of a
        hundred generations builds one of these per candidate.
        """
        # One device, named: ``devices="auto"`` on a machine with several GPUs
        # would spread one candidate over all of them, and the search evaluates
        # thousands of candidates one after another.
        cuda = device.type == "cuda"
        return L.Trainer(
            max_epochs=self.predictor_epochs,
            accelerator="gpu" if cuda else device.type,
            # One, explicitly: this trainer is already inside a worker, and
            # a candidate spread over more of them would fight its siblings.
            devices=[self._cuda_index(device)] if cuda else 1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )

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
            batches = (
                _Batches(train_loader)
                if isinstance(train_loader, Sequence)
                else train_loader
            )
            self._lightning(device).fit(model, train_dataloaders=batches)
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
                valid = batch.mask.sum(dim=-1).clamp_min(1)
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
        weights = [individual.fitness for individual in self.population]
        children = []
        for _ in range(self.population_size // 2):
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
        point = self._random.randrange(chromosome_1.numel())
        return (
            th.cat((chromosome_1[:point], chromosome_2[point:])),
            th.cat((chromosome_2[:point], chromosome_1[point:])),
        )

    def _mutate(self, chromosome: th.Tensor) -> th.Tensor:
        selected = (
            th.rand(chromosome.shape, generator=self._torch_generator)
            < self.mutation_probability
        )
        noise = th.randn(chromosome.shape, generator=self._torch_generator)
        return chromosome + selected.to(chromosome.dtype) * noise * self.mutation_std

    @staticmethod
    def _chromosome(model: GenSPP) -> th.Tensor:
        return th.nn.utils.parameters_to_vector(model.generator_parameters()).detach()
