import math
import random
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, Registry
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import (
    GenSPP,
    GenSPPTrainer,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
    genspp,
)
from pyhighlights.components.models.spp.genspp import _Individual
from pyhighlights.configurations.keys import GRU_GENSPP, GRU_GENSPP_TRAINER

NAMESPACE = "tests_genspp"


class TinyBackbone(SPPBackbone):
    def __init__(self):
        super().__init__()
        self.embedding = th.nn.Embedding(8, 3)

    @property
    def output_size(self) -> int:
        return 3

    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        selected = mask if selection_mask is None else mask * selection_mask
        return self.embedding(features) * selected.unsqueeze(-1)

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        float_mask = mask.unsqueeze(-1)
        return (states * float_mask).sum(1) / float_mask.sum(1).clamp_min(1)


class TinySelector(SPPSelector):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = th.nn.Linear(input_size, 2)

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.linear(states)


class TinyPredictor(SPPPredictor):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = th.nn.Linear(input_size, 2)

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.linear(states)


class CountingGenSPP(GenSPP):
    instances = 0

    def __init__(self, **kwargs):
        type(self).instances += 1
        super().__init__(**kwargs)


class ClassificationLossConfig(Configuration):
    name: str = Param("classification")
    loss: RegistrationKey = Param(
        RegistrationKey(name="criterion", namespace=NAMESPACE)
    )
    inputs: List[str] = Param(["class_logits", "y_true"])


class TinyGenSPPConfig(Configuration):
    name: str = Param("genspp")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(
        RegistrationKey(name="backbone", namespace=NAMESPACE)
    )
    selectors: RegistrationKey[SPPSelector] = Param(
        RegistrationKey(name="selector", namespace=NAMESPACE)
    )
    predictor: RegistrationKey[SPPPredictor] = Param(
        RegistrationKey(name="predictor", namespace=NAMESPACE)
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        RegistrationKey(name="backbone", namespace=NAMESPACE)
    )
    aggregator: RegistrationKey | None = Param(None)
    temperature: float = Param(1.0)
    losses: List[RegistrationKey] = Param(
        [RegistrationKey(name="loss", namespace=NAMESPACE)]
    )
    optimizer: RegistrationKey = Param(
        RegistrationKey(name="optimizer", namespace=NAMESPACE)
    )
    train_metrics: List[RegistrationKey] | None = Param(None)
    val_metrics: List[RegistrationKey] | None = Param(None)
    test_metrics: List[RegistrationKey] | None = Param(None)


class AdamConfig(Configuration):
    lr: float = Param(1e-3)


def register(name: str, component: str) -> RegistrationKey:
    return Registry.register_configuration(
        config=Configuration.default(),
        name=name,
        namespace=NAMESPACE,
        component=component,
    )


def register_tiny_genspp() -> RegistrationKey:
    Registry.initialize()
    CountingGenSPP.instances = 0
    backbone = register("backbone", f"{__name__}.TinyBackbone")
    selector = register("selector", f"{__name__}.TinySelector")
    predictor = register("predictor", f"{__name__}.TinyPredictor")
    register("criterion", "torch.nn.CrossEntropyLoss")
    Registry.register_configuration(
        config=ClassificationLossConfig.default(),
        name="loss",
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    optimizer = Registry.register_configuration(
        config=AdamConfig.default(),
        name="optimizer",
        namespace=NAMESPACE,
        component="torch.optim.Adam",
    )
    config = TinyGenSPPConfig.default()
    config.selector_backbones = backbone
    config.selectors = selector
    config.predictor = predictor
    config.predictor_backbone = backbone
    config.optimizer = optimizer
    model = Registry.register_configuration(
        config=config,
        name="model",
        tags={"genspp"},
        namespace=NAMESPACE,
        component=f"{__name__}.CountingGenSPP",
    )
    Registry.dag_resolution()
    return model


def batch(sample_ids=(0, 1), labels=(0, 1)) -> InputData:
    return InputData(
        features=th.tensor([[1, 2, 3], [4, 5, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        sample_ids=th.tensor(sample_ids),
        y_true=th.tensor(labels),
        highlight_true=th.full((2, 3), -1),
    )


def trainer(model: RegistrationKey, **kwargs) -> GenSPPTrainer:
    parameters = {
        "n_generations": 0,
        "population_size": 2,
        "predictor_epochs": 1,
        "task_loss_limit": 10.0,
        "seed": 7,
        **kwargs,
    }
    return GenSPPTrainer(model=model, **parameters)


def test_registered_gru_genspp_and_trainer():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_GENSPP)
    search = Registry.from_key(GRU_GENSPP_TRAINER)

    assert isinstance(model, GenSPP)
    assert isinstance(search, GenSPPTrainer)
    assert search.model == GRU_GENSPP
    assert model.selector_backbone is not model.predictor_backbone
    assert model.selector_backbone.encoder.bidirectional is False
    assert model.selector_backbone.output_size == 16
    assert model.selector_backbone.embedding.weight.requires_grad is False
    assert th.equal(
        model.selector_backbone.embedding.weight,
        model.predictor_backbone.embedding.weight,
    )
    second_model = Registry.from_key(GRU_GENSPP)
    assert not th.equal(
        model.selector_backbone.embedding.weight,
        second_model.selector_backbone.embedding.weight,
    )
    search._align_initial_state(model)
    search._align_initial_state(second_model)
    assert th.equal(
        model.selector_backbone.embedding.weight,
        second_model.selector_backbone.embedding.weight,
    )
    # The predictor too, and that is the part a candidate's fitness depends on:
    # two candidates must train the same predictor or they are not comparable.
    for first, second in zip(
        model.predictor_parameters(), second_model.predictor_parameters()
    ):
        assert th.equal(first, second)
    optimizer = Registry.from_key(model.optimizer, params=model.predictor_parameters())
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-2)
    # Lightning trains one candidate at a time, and only its predictor.
    configured = model.configure_optimizers()
    assert [id(parameter) for parameter in configured.param_groups[0]["params"]] == [
        id(parameter) for parameter in model.predictor_parameters()
    ]


def test_genspp_is_independent_and_permits_empty_highlights():
    model = Registry.from_key(register_tiny_genspp())
    assert isinstance(model, GenSPP)
    assert model.selector_backbone is not model.predictor_backbone

    with th.no_grad():
        model.selectors[0].linear.weight.zero_()
        model.selectors[0].linear.bias.copy_(th.tensor([2.0, -2.0]))

    model.train()
    first = model(batch())
    second = model(batch())

    assert first.class_logits.shape == (2, 1, 2)
    assert first.highlight_logits.shape == (2, 1, 3, 2)
    assert first.highlight_mask.shape == (2, 1, 3)
    assert not first.highlight_mask.any()
    assert th.isfinite(first.class_logits).all()
    assert th.equal(first.highlight_mask, second.highlight_mask)


def test_inner_training_changes_only_predictor_and_validation_scores_fitness():
    model_key = register_tiny_genspp()
    model = Registry.from_key(model_key)
    search = trainer(model_key)
    generator_before = [
        parameter.detach().clone() for parameter in model.generator_parameters()
    ]
    predictor_before = [
        parameter.detach().clone() for parameter in model.predictor_parameters()
    ]

    cpu = th.device("cpu")
    search._train_predictor(model, [batch(sample_ids=(10, 11), labels=(0, 0))], cpu)

    assert all(
        th.equal(before, after)
        for before, after in zip(generator_before, model.generator_parameters())
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(predictor_before, model.predictor_parameters())
    )

    validation = batch(sample_ids=(20, 21), labels=(1, 1))
    task_loss, selection_rate = search._evaluate(model, [validation], cpu)
    with th.no_grad():
        output = model(validation)
        expected_loss = th.nn.functional.cross_entropy(
            output.class_logits[:, 0], validation.y_true
        ).item()
        expected_rate = (
            (output.highlight_mask[:, 0].sum(-1) / validation.mask.sum(-1).clamp_min(1))
            .mean()
            .item()
        )
    assert task_loss == pytest.approx(expected_loss)
    assert selection_rate == pytest.approx(expected_rate)


def test_a_candidate_scores_the_same_wherever_it_is_evaluated():
    """Fitness is a property of the chromosome, not of evaluation order.

    Before this held, the same chromosome scored 2.8246, 2.4722 and 1.0000 as
    the first, second and fourth candidate of one run: every evaluation built
    a predictor from the global random state, so each one shifted the next.
    A search over such a fitness ranks initialisations alongside genes, and
    parallel evaluation could not reproduce a sequential run at all.
    """
    model = register_tiny_genspp()
    search = trainer(model, seed=7)
    search._random.seed(7)
    search._torch_generator.manual_seed(7)
    search._initial_state = None
    train, val = [batch()], [batch()]

    probe = search._chromosome(Registry.from_key(model)).clone()
    decoy = probe + 0.3

    scores = []
    for position in range(3):
        for _ in range(position):
            search._score(train, val, [decoy])
        scores.append(search._score(train, val, [probe])[0].fitness)

    assert scores[0] == scores[1] == scores[2]


def test_the_first_generation_is_not_one_point_repeated():
    """Sharing a predictor between candidates must not share their generators.

    The shared initial state is what makes fitness comparable, and it covers
    everything except the chromosome. Covering the chromosome too would give
    every founder the same generator -- a first generation of one repeated
    point, with mutation left as the only source of diversity.
    """
    model = register_tiny_genspp()
    search = trainer(model, seed=7, population_size=4)
    search.fit([batch()], [batch()])

    chromosomes = [individual.chromosome for individual in search.population]
    assert len(chromosomes) == 4
    for other in chromosomes[1:]:
        assert not th.equal(chromosomes[0], other)


def test_scoring_a_candidate_does_not_read_the_global_random_state():
    """The property several devices depend on.

    torch's default generator is process-wide, so anything a worker drew from
    it would depend on how the threads interleaved. No result may: the model's
    own initialisation is overwritten by `_initial_state` outside the
    chromosome and by the chromosome inside it, the batches arrive as the list
    `_fit` froze rather than as a loader that would shuffle them again, and the
    registered configurations set `dropout_rate` to 0. A dropout rate above
    zero would take that back silently, which is why this is asserted rather
    than assumed.

    Scoring does *advance* that state -- building a model draws from it -- and
    that is fine, which is the second assertion here: the search's own
    randomness lives on explicit generators, so nothing reads what scoring
    left behind.
    """
    model = register_tiny_genspp()
    search = trainer(model, seed=7, task_loss_limit=10.0)
    train, val = [batch(labels=(0, 1))], [batch(labels=(0, 1))]
    probe = search._founder_chromosome()

    before = search._score(train, val, [probe])[0]
    th.rand(1000)
    after = search._score(train, val, [probe])[0]

    assert before.fitness == after.fitness
    assert before.task_loss == after.task_loss

    # It advances the state, so a test asserting otherwise would be wrong.
    state = th.get_rng_state().clone()
    search._score(train, val, [probe])
    assert not th.equal(state, th.get_rng_state())


def test_several_devices_give_the_same_population_as_one():
    """Threads change when a candidate is scored, never what it scores."""
    model = register_tiny_genspp()
    train, val = [batch(labels=(0, 1))], [batch(labels=(0, 1))]

    # Eight candidates against four workers, so several are genuinely in
    # flight at once: a population of two would leave the pool half idle and
    # a race unexercised.
    sequential = trainer(
        model, n_generations=1, population_size=8, task_loss_limit=10.0
    )
    sequential.fit(train, val)

    parallel = trainer(
        model,
        n_generations=1,
        population_size=8,
        task_loss_limit=10.0,
        devices=["cpu"] * 4,
    )
    parallel.fit(train, val)

    assert sequential.training_progress == parallel.training_progress
    assert [individual.fitness for individual in sequential.population] == [
        individual.fitness for individual in parallel.population
    ]


def test_every_candidate_trains_on_the_same_batch_order():
    """A shuffling loader must not make fitness depend on evaluation order.

    `GenSPPTask` hands the search the training `DataLoader` the task built,
    which shuffles. Re-iterating it draws a new permutation, so before `_fit`
    froze one the same chromosome scored differently depending on how many
    candidates preceded it -- and with a pool, on how the workers interleaved.
    """
    orders: List[List[List[int]]] = []

    class RecordingTrainer(GenSPPTrainer):
        def _train_predictor(self, model, train_loader, device):
            orders.append([list(item.sample_ids.tolist()) for item in train_loader])
            super()._train_predictor(model, train_loader, device)

    rows = [
        InputData(
            features=th.tensor([[1, 2, 3]]),
            mask=th.tensor([[1.0, 1.0, 1.0]]),
            sample_ids=th.tensor([index]),
            y_true=th.tensor([index % 2]),
            highlight_true=th.full((1, 3), -1),
        )
        for index in range(8)
    ]
    shuffling = DataLoader(rows, batch_size=None, shuffle=True)

    search = RecordingTrainer(
        model=register_tiny_genspp(),
        n_generations=1,
        population_size=4,
        predictor_epochs=1,
        task_loss_limit=10.0,
        seed=7,
    )
    search.fit(shuffling, rows)

    # One founder and one child per candidate of a two-generation search, so
    # this is not one candidate compared with itself.
    assert len(orders) == 8
    assert all(order == orders[0] for order in orders[1:])
    # The frozen order is one the loader shuffled, not the order it was
    # written in: a loader that happened not to shuffle would pass the
    # assertion above without proving anything.
    assert orders[0] != [[index] for index in range(8)]


def test_a_diverged_candidate_is_refused_where_it_diverged():
    """A NaN loss must not be found a generation later by the roulette wheel.

    `compute_fitness` compares the loss against its limit, and a NaN compares
    false, so the candidate came back with a NaN fitness instead of the floor.
    That survives into the population and fails inside `random.choices` with
    `Total of weights must be finite` -- which names neither the candidate nor
    the device that trained it.
    """
    assert math.isnan(
        GenSPPTrainer.compute_fitness(
            task_loss=float("nan"), selection_rate=0.3, task_loss_limit=10.0
        )
    )

    search = trainer(register_tiny_genspp())
    search._evaluate = lambda *arguments: (float("nan"), 0.3)

    with pytest.raises(ValueError, match="non-finite fitness"):
        search.fit([batch()], [batch()])


def test_search_rejects_single_pass_loaders():
    search = trainer(register_tiny_genspp())
    with pytest.raises(ValueError, match="re-iterable"):
        search.fit(iter([batch()]), [batch()])


def test_fitness_genetic_operators_and_survival_match_contract():
    valid = GenSPPTrainer.compute_fitness(0.25, 0.36, 0.25)
    expected = 1.0 / (1.0 - math.sqrt(0.64 * 0.75) + 1e-8)
    assert valid == pytest.approx(expected)
    assert GenSPPTrainer.compute_fitness(0.2501, 0.0, 0.25) == 1.0

    search = trainer(register_tiny_genspp(), mutation_probability=1.0)
    first = th.arange(5, dtype=th.float32)
    second = -first
    search._random.seed(3)
    child_1, child_2 = search._crossover(first, second)
    point = random.Random(3).randrange(first.numel())
    assert th.equal(child_1, th.cat((first[:point], second[point:])))
    assert th.equal(child_2, th.cat((second[:point], first[point:])))

    search._torch_generator.manual_seed(5)
    mutated = search._mutate(first)
    expected_generator = th.Generator().manual_seed(5)
    th.rand(first.shape, generator=expected_generator)
    expected_noise = th.randn(first.shape, generator=expected_generator)
    assert th.allclose(mutated, first + expected_noise * search.mutation_std)

    candidates = [SimpleNamespace(fitness=float(value)) for value in range(8)]
    survivors = search._select_survivors(candidates)
    assert len(survivors) == search.population_size
    assert candidates[-1] in survivors


def test_search_is_reproducible_and_keeps_only_candidate_chromosomes():
    model_key = register_tiny_genspp()
    train = [batch(labels=(0, 0))]
    validation = [batch(labels=(1, 1))]

    first_search = trainer(model_key, n_generations=1)
    first_model = first_search.fit(train, validation)
    # Two founders and two children scored, one build per founder to draw its
    # generator at random -- scoring cannot draw it, since no result there may
    # read the global random state -- one more to capture the shared initial
    # state before any pool exists, and one per candidate that turned out to
    # be the best so far, which the search rebuilds from its chromosome and
    # the weights descent left on it. Everything but the four scorings is paid
    # per run or per improvement rather than per candidate.
    assert CountingGenSPP.instances == 9
    first_state = {
        name: value.detach().clone() for name, value in first_model.state_dict().items()
    }

    second_search = trainer(model_key, n_generations=1)
    second_model = second_search.fit(train, validation)

    assert first_search.training_progress == second_search.training_progress
    assert all(
        th.equal(value, second_model.state_dict()[name])
        for name, value in first_state.items()
    )
    assert len(first_search.population) == 2
    assert all(
        not hasattr(individual, "model") for individual in first_search.population
    )
    assert (
        first_search.population[0].chromosome.data_ptr()
        != first_search.population[1].chromosome.data_ptr()
    )


def test_a_generation_draws_couples_and_keeps_the_population_whole():
    """``selection_rate`` counts couples, not children, and not survivors.

    The release computes ``n_couples = int(selection_rate * len(population))``
    and crosses each couple into two, so its default 0.5 adds one child per
    member: 25 couples and 50 children for a population of 50. This was
    written as ``population_size // 2``, which is the same number with the
    rate baked in -- and reads as though a generation bred half a population.

    What has to hold whatever the rate is: the population that comes out is
    the population that went in, because survival keeps exactly
    ``population_size``.
    """
    model = register_tiny_genspp()
    train, val = [batch(labels=(0, 1))], [batch(labels=(0, 1))]

    for rate, couples in ((0.5, 4), (1.0, 8), (0.25, 2)):
        search = trainer(model, n_generations=2, population_size=8, selection_rate=rate)
        search.fit(train, val)
        assert int(rate * 8) == couples
        assert len(search.population) == 8

    # The elites lead and every drawn survivor sits below them. The drawn half
    # is in the order the sampler returned, not in fitness order, so the list
    # as a whole is not sorted.
    fitnesses = [individual.fitness for individual in search.population]
    assert fitnesses[:4] == sorted(fitnesses, reverse=True)[:4]
    assert min(fitnesses[:4]) >= max(fitnesses[4:])


def test_a_rate_that_draws_no_couple_is_refused():
    """A generation with no children is a search that cannot move."""
    model = register_tiny_genspp()
    with pytest.raises(ValueError, match="draws no couple"):
        trainer(model, population_size=8, selection_rate=0.1)
    with pytest.raises(ValueError, match="selection_rate must be in"):
        trainer(model, selection_rate=0.0)
    with pytest.raises(ValueError, match="selection_rate must be in"):
        trainer(model, selection_rate=1.5)


def test_survivors_are_half_elites_and_half_drawn_without_replacement():
    """Half elitism, which is what keeps a hundred generations from collapsing.

    ``HalfElitismSurvival`` in the release keeps the best half outright and
    draws the other half from everything below, fitness-proportional and
    without replacement. Without the second half the search is hill climbing
    on one lineage; with replacement a single chromosome could take every
    remaining place.
    """
    model = register_tiny_genspp()
    search = trainer(model, population_size=8)
    candidates = [
        _Individual(
            chromosome=th.full((1,), float(index)),
            fitness=float(index + 1),
            task_loss=0.0,
            selection_rate=0.0,
        )
        for index in range(16)
    ]

    survivors = search._select_survivors(candidates)

    assert len(survivors) == 8
    # The four best are kept because they are the four best.
    assert [individual.fitness for individual in survivors[:4]] == [
        16.0,
        15.0,
        14.0,
        13.0,
    ]
    # The other four come from below them, and none is kept twice.
    drawn = [individual.fitness for individual in survivors[4:]]
    assert len(set(drawn)) == 4
    assert all(fitness <= 12.0 for fitness in drawn)


def subword_batch() -> InputData:
    """Five subtokens spelling two words in one row and one in the other.

    ``word_ids`` is what separates the two axes, and a subword tokenizer is
    the only thing that produces a batch where they differ.
    """
    return InputData(
        features=th.tensor([[7, 1, 2, 3, 6], [7, 4, 6, 0, 0]]),
        mask=th.tensor([[1.0, 1.0], [1.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 2), -1),
        word_ids=th.tensor([[-1, 0, 0, 1, -1], [-1, 0, -1, -1, -1]]),
        attention_mask=th.tensor(
            [[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0, 0.0]]
        ),
    )


def test_the_selection_rate_is_measured_on_the_axis_the_selection_was_made_on():
    """A subtoken selection is a share of the subtokens, not of the words.

    ``mask`` is the word axis whatever a model selects over, so dividing a
    subtoken count by it reports a rate the search then optimises: this batch
    has three selectable subtokens against two words in its first row, so the
    old denominator could put the rate above one.
    """
    model_key = register_tiny_genspp()
    model = Registry.from_key(model_key, expected_type=GenSPP)
    model.select_over = "subtoken"
    search = trainer(model_key)
    data = subword_batch()

    _, rate = search._evaluate(model, [data], th.device("cpu"))

    model.eval()
    with th.no_grad():
        selected = model(data).highlight_mask[:, 0]
    selectable = (data.word_ids >= 0).float()
    expected = ((selected * selectable).sum(dim=-1) / selectable.sum(dim=-1)).mean()

    assert rate == pytest.approx(float(expected))
    assert 0.0 <= rate <= 1.0


def test_a_diagnosed_search_scores_one_candidate_at_a_time(caplog, monkeypatch):
    """Threads interleave, and the record is attributed by order alone.

    Two candidates writing their stages at once read as one model that never
    existed, so a search asked to diagnose itself takes the sequential path
    however many devices it was given -- and says which candidate each run of
    lines belongs to.
    """
    import logging

    from pyhighlights.components.models.spp import genspp as module
    from pyhighlights.utility import diagnostics

    model = register_tiny_genspp()
    train, val = [batch(labels=(0, 1))], [batch(labels=(0, 1))]
    search = trainer(
        model,
        n_generations=1,
        population_size=2,
        task_loss_limit=10.0,
        devices=["cpu"] * 2,
    )

    def refuse(*args, **kwargs):
        raise AssertionError("a diagnosed search opened a pool")

    monkeypatch.setattr(module, "ThreadPool", refuse)
    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        search.fit(train, val)

    assert "candidate: index = 0" in caplog.text
    assert "candidate: index = 1" in caplog.text
    assert "generation: index = 0" in caplog.text


def test_the_winner_says_its_generator_is_not_trained():
    """It is a chromosome the search settled on, and nothing moves it again.

    Scoring the winner is a forward pass and a second search draws its own
    founders, so a generator left marked trainable reads as a model half of
    which descent produced -- to a cost table counting parameters, and to
    anything building an optimizer over `parameters()`.
    """
    model_key = register_tiny_genspp()
    search = trainer(model_key, n_generations=1)

    model = search.fit([batch(labels=(0, 1))], [batch(labels=(0, 1))])

    assert not model.generator_parameters()
    assert model.predictor_parameters()
    # The weights themselves are untouched: what changed is what is said
    # about them.
    assert any(
        parameter.numel()
        for parameter in chain(
            model.selector_backbones.parameters(), model.selectors.parameters()
        )
    )


def test_a_candidate_comes_back_as_a_chromosome_and_what_descent_moved():
    """A worker in another process cannot hand a model back over a pipe.

    So it hands back neither: the chromosome it was given, and the weights
    gradient descent left on the predictor. Everything frozen is left out --
    a GloVe table is megabytes, is the same in every candidate, and is loaded
    from the search's own copy when the model is built again.
    """
    search = trainer(register_tiny_genspp(), devices=["cpu"])
    search._embeddings = None
    search._initial_state = None
    search._candidate()
    chromosome = search._founder_chromosome()

    individual, scored = search._evaluate_individual(
        [batch()], [batch()], chromosome, th.device("cpu")
    )
    state = search._trained_state(scored)
    restored = search._restored(individual.chromosome, state)

    assert all(
        th.equal(value, restored.state_dict()[name])
        for name, value in scored.state_dict().items()
    )

    # And what is frozen is left out, which on a real corpus is the embedding
    # table. This model freezes nothing, so one is frozen here to say so.
    model = search._candidate()
    name, parameter = next(iter(model.named_parameters()))
    parameter.requires_grad_(False)

    assert name not in search._trained_state(model)


def _refuses_to_differentiate() -> bool:
    """What a worker does when torch has closed fork to this process."""
    raise RuntimeError(
        "Unable to handle autograd's threading in combination with "
        "fork-based multiprocessing."
    )


def test_a_pool_that_cannot_differentiate_is_not_used(monkeypatch):
    """Torch refuses fork once autograd has run threads in the parent.

    It refuses in the worker rather than at the fork, so a search that asked
    no questions would lose a whole generation to it. This one asks, and a
    pool that cannot answer is closed and replaced by the threads the search
    used to use.

    The refusal is forced here rather than provoked: whether torch has started
    autograd's threads depends on what else has run in the process, so a test
    that trained something first would assert it on some runs and not others.
    """
    monkeypatch.setattr(genspp, "_worker_can_train", _refuses_to_differentiate)
    model = register_tiny_genspp()
    train, validation = [batch(labels=(0, 1))], [batch(labels=(0, 1))]

    search = trainer(model, devices=["cpu"] * 4)
    search._open_pool(train, validation)

    assert search._pool is None
    assert genspp._WORK is None
    # And the search still runs, on threads.
    assert search.fit(train, validation) is not None


def test_a_frozen_predictor_backbone_does_not_drop_while_a_candidate_trains():
    """Or a chromosome's fitness is a property of the random state too.

    `GenSPPTransformerBackboneConfig` sets `freeze_transformer`, and a frozen
    Hugging Face encoder left in training mode still drops: the same candidate
    would score differently depending on what had drawn before it, which
    across workers is a matter of scheduling. Only the generator used to be
    put back into eval mode, because only the generator is frozen when the
    backbone is a GRU this study trains.
    """
    search = trainer(register_tiny_genspp(), devices=["cpu"])
    search._embeddings = None
    search._initial_state = None
    model = search._candidate()
    for parameter in model.predictor_backbone.parameters():
        parameter.requires_grad_(False)

    # What `_train_predictor` does before it steps the model.
    model.train(True)
    model.on_train_epoch_start()

    assert not model.predictor_backbone.training
    # The predictor itself is what descent moves, and does train.
    assert model.predictor.training


def test_two_unseeded_searches_do_not_draw_the_same_founders():
    """`seed=None` means this run should differ from the last one.

    Founders are built from the global generator, and `fit` restores that
    around the whole search -- so a search that reseeded only its own
    generators drew the same population every time, and differed after that
    only in selection and mutation.
    """
    model = register_tiny_genspp()
    train, validation = [batch(labels=(0, 1))], [batch(labels=(0, 1))]

    searches = []
    for _ in range(2):
        search = trainer(model, seed=None, n_generations=0, task_loss_limit=10.0)
        search.fit(train, validation)
        searches.append(search)

    first, second = (search.population for search in searches)
    assert not any(
        th.equal(one.chromosome, other.chromosome) for one, other in zip(first, second)
    )


def _refuses_to_fork() -> bool:
    """A pool that fails for something other than autograd's threading."""
    raise OSError("Cannot allocate memory")


def test_a_pool_that_fails_to_launch_leaves_nothing_behind(monkeypatch):
    """Not every launch failure is the `RuntimeError` autograd raises.

    A node out of memory or file descriptors raises `OSError` instead, and a
    worker that never answers raises neither. Whatever the reason, the search
    is left with no pool, no workers and nothing holding the corpus it was
    about to hand them.
    """
    monkeypatch.setattr(genspp, "_worker_can_train", _refuses_to_fork)
    model = register_tiny_genspp()
    train, validation = [batch(labels=(0, 1))], [batch(labels=(0, 1))]

    search = trainer(model, devices=["cpu"] * 4)
    search._open_pool(train, validation)

    assert search._pool is None
    assert genspp._WORK is None
    assert search.fit(train, validation) is not None
