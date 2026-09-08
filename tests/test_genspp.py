import math
import random
from pathlib import Path
from types import SimpleNamespace
from typing import List

import pytest
import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import (
    GenSPP,
    GenSPPTrainer,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
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
    search._align_frozen_state(model)
    search._align_frozen_state(second_model)
    assert th.equal(
        model.selector_backbone.embedding.weight,
        second_model.selector_backbone.embedding.weight,
    )
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

    search._train_predictor(model, [batch(sample_ids=(10, 11), labels=(0, 0))])

    assert all(
        th.equal(before, after)
        for before, after in zip(generator_before, model.generator_parameters())
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(predictor_before, model.predictor_parameters())
    )

    validation = batch(sample_ids=(20, 21), labels=(1, 1))
    task_loss, selection_rate = search._evaluate(model, [validation])
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
    assert CountingGenSPP.instances == 4
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
