import inspect
import math
from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import RegistrationKey, Registry
from pydantic import BaseModel, ValidationError
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp import FR, MCD, MGR
from pyhighlights.configurations import keys
from pyhighlights.configurations.evalx import RandomSelectorConfig
from pyhighlights.configurations.genspp import (
    GRUGenSPPTrainerConfig,
    ToyGenSPPTrainerConfig,
)
from pyhighlights.configurations.grat import GRUGRATConfig
from pyhighlights.configurations.keys import (
    ADAM,
    GRU_BACKBONE,
    GRU_DAR,
    GRU_FR,
    GRU_GRAT,
    GRU_MCD,
    GRU_MGR,
    GUIDANCE_GUIDER,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_FR,
    TRANSFORMER_GRAT,
    TRANSFORMER_GUIDER,
)
from pyhighlights.configurations.losses import (
    CrossEntropyConfig,
    MaskedCrossEntropyConfig,
)
from pyhighlights.configurations.metrics import ClassF1Config, HighlightMetricConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights.configurations.tasks import TaskConfig, ToyTaskConfig
from pyhighlights.utility.spaces import token_spaces


def test_every_registration_in_the_library_resolves():
    """No key is dropped as invalid, and each one builds.

    A condition that reads a field wrongly, or a key referencing one that was
    never registered, is a whole experiment silently missing from a grid rather
    than an error. Five components are built with a run-time argument, since
    their input size is measured from a backbone and their parameters from a
    built model. Neither is knowable before the run, so neither is a
    parameter.
    """
    valid, invalid = Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert not invalid, sorted(str(key) for key in invalid)
    assert valid

    runtime = {"input_size": 6, "params": [th.nn.Parameter(th.zeros(1))]}
    needed_runtime = set()
    for key in valid:
        try:
            built = Registry.from_key(key)
            # Every name a registered model logs is checked here, since
            # otherwise the check runs only for the models a test trains.
            if isinstance(built, Model):
                built.setup("fit")
            continue
        except ImportError as error:
            # The optional extra reporting itself, which is the whole of what a
            # transformer backbone needs that this environment may not have.
            assert "pyhighlights[transformers]" in str(error)
            continue
        except TypeError as error:
            missing = error
        for name, value in runtime.items():
            try:
                Registry.from_key(key, **{name: value})
            except TypeError:
                continue
            needed_runtime.add(name)
            break
        else:
            raise missing

    assert needed_runtime == {"input_size", "params"}


def test_every_declared_key_is_registered():
    """Each key in ``keys.py`` names a registration the namespace scan finds.

    A study that builds the library as an external directory loads it only
    through the namespace ``Registry.build`` reads off its sources before
    importing anything. The scan follows ``key=`` only to a ``Namespace``
    literal, so a key computed any other way leaves the library unreachable.
    """
    directory = Path(pyhighlights.__file__).parent
    _, mapping = Registry.parse_configuration_files([directory])
    assert mapping.get(keys.NAMESPACE) == directory

    Registry.build(directory=directory)
    declared = {
        name: value
        for name, value in vars(keys).items()
        if isinstance(value, RegistrationKey)
    }

    assert declared
    assert not [name for name, key in declared.items() if not Registry.in_registry(key)]


def test_registered_gru_fr_forward_backward_and_optimizer():
    # Seeded before the model is built, because an unrepaired sample can drop
    # every word, and an empty selection pools to zeros and gives the
    # predictor's weight no gradient. Seeding after the build left the weights
    # to whatever state earlier tests in the worker left the generator in.
    th.manual_seed(0)
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_FR)
    assert isinstance(model, FR)
    # The original FR's rules: a sampled selection at evaluation and no
    # empty-selection repair.
    assert model.sample_at_evaluation
    assert not model.repair_empty_selections

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )

    model.train()
    output = model(batch)
    loss, losses = model.compute_loss(batch, output)
    loss.backward()

    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_logits.shape == (2, 1, 4, 2)
    assert output.highlight_mask.shape == (2, 1, 4)
    assert set(losses) == {"classification", "sparsity", "contiguity"}
    assert model.selector_backbone.embedding.weight.grad is not None
    assert model.selectors[0].selector[0].weight.grad is not None
    predictor_weight = model.predictor.predictor[-1].weight
    assert predictor_weight.grad is not None
    assert predictor_weight.grad.abs().sum() > 0

    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, th.optim.Adam)
    previous_weight = predictor_weight.detach().clone()
    optimizer.step()
    assert not th.equal(previous_weight, predictor_weight)

    # The argmax, so two passes select alike.
    model.sample_at_evaluation = False
    model.eval()
    with th.no_grad():
        first = model(batch).highlight_mask
        second = model(batch).highlight_mask
        short_states = model.selector_backbone.encode(
            th.tensor([[1, 2, 3]]), th.ones((1, 3))
        )
        padded_states = model.selector_backbone.encode(
            th.tensor([[1, 2, 3, 0, 0]]),
            th.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        )
    assert th.equal(first, second)
    assert th.allclose(short_states, padded_states[:, :3], atol=1e-6)


def test_registered_gru_mcd_alternates_generator_and_predictor_updates():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_MCD)
    assert isinstance(model, MCD)
    assert model.automatic_optimization is False
    assert model.selector_backbone is not model.predictor_backbone

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )
    output = model(batch)
    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_logits.shape == (2, 1, 4, 2)

    for loss in model.shared_losses:
        loss.enabled = False
    predictor_total, predictor_losses, _ = model.predictor_phase_loss(batch)
    assert set(predictor_losses) == {"classification", "full_classification"}
    predictor_total.backward()
    assert all(
        parameter.grad is None for parameter in model.selector_backbone.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())

    model.zero_grad(set_to_none=True)
    for loss in model.shared_losses:
        loss.enabled = True
    generator_total, generator_losses, _ = model.generator_phase_loss(batch)
    assert set(generator_losses) == {"sparsity", "contiguity", "discrepancy"}
    generator_total.backward()
    assert any(
        parameter.grad is not None for parameter in model.selector_backbone.parameters()
    )

    model = Registry.from_key(GRU_MCD)
    generator_parameters = [
        *model.selector_backbones.parameters(),
        *model.selectors.parameters(),
    ]
    predictor_parameters = [
        *model.predictor_backbone.parameters(),
        *model.predictor.parameters(),
    ]
    generator_before = [
        parameter.detach().clone() for parameter in generator_parameters
    ]
    predictor_before = [
        parameter.detach().clone() for parameter in predictor_parameters
    ]
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))
    assert any(
        not th.equal(before, after)
        for before, after in zip(generator_before, generator_parameters)
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(predictor_before, predictor_parameters)
    )


def test_registered_gru_mgr_has_independent_generators_and_head_policy():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_MGR, sample_at_evaluation=False)
    assert isinstance(model, MGR)
    assert len(model.selectors) == len(model.selector_backbones) == 3
    assert len({id(backbone) for backbone in model.selector_backbones}) == 3
    assert len({id(selector) for selector in model.selectors}) == 3
    assert all(
        model.predictor_backbone is not backbone
        for backbone in model.selector_backbones
    )

    optimizer = model.configure_optimizers()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-3 / 3, 1e-3, 2e-3, 3e-3]
    )
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter)
        for parameter in [
            *model.predictor_backbone.parameters(),
            *model.predictor.parameters(),
        ]
    }
    for index, (backbone, selector) in enumerate(
        zip(model.selector_backbones, model.selectors), start=1
    ):
        assert {
            id(parameter) for parameter in optimizer.param_groups[index]["params"]
        } == {
            id(parameter)
            for parameter in [*backbone.parameters(), *selector.parameters()]
        }

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )

    model.train()
    output = model(batch)
    summed_loss, summed_parts = model.compute_loss(batch, output)
    model.loss_reduction = "mean"
    mean_loss, mean_parts = model.compute_loss(batch, output)

    assert output.class_logits.shape == (2, 3, 2)
    assert output.highlight_logits.shape == (2, 3, 4, 2)
    assert output.highlight_mask.shape == (2, 3, 4)
    assert th.allclose(mean_loss * 3, summed_loss)
    assert all(
        th.allclose(mean_parts[name] * 3, summed_parts[name]) for name in summed_parts
    )

    mean_loss.backward()
    assert all(
        backbone.embedding.weight.grad is not None
        for backbone in model.selector_backbones
    )
    assert model.predictor_backbone.embedding.weight.grad is not None

    model.inference_head = 1
    model.eval()
    with th.no_grad():
        expected = model.forward_one_head(batch, selector_idx=1)
        actual = model.validation_forward(batch)
    assert actual.class_logits.shape == (2, 1, 2)
    assert th.equal(actual.class_logits, expected.class_logits)
    assert th.equal(actual.highlight_mask, expected.highlight_mask)


def test_mgr_scores_the_inference_head_and_not_the_first_one(monkeypatch):
    """Which generator a training metric is about.

    MGR emits every head while training, and `update_metrics` reads
    `inference_head` through `SPP.reported` before scoring. A wrong index here
    would report generator 0's highlight through training while validation
    reported generator 1's, which gives two disagreeing numbers about one
    model.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_MGR)
    model.inference_head = 2
    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )

    model.eval()
    with th.no_grad():
        output = model(batch)
    assert output.class_logits.shape[1] == 3

    seen = {}
    monkeypatch.setattr(
        Model,
        "update_metrics",
        lambda self, split, input_data, output_data: seen.update(output=output_data),
    )
    model.update_metrics("train", batch, output)

    scored = seen["output"]
    assert scored.class_logits.shape == (2, 2)
    assert th.equal(scored.class_logits, output.class_logits[:, 2])
    assert th.equal(scored.highlight_mask, output.highlight_mask[:, 2])


def test_a_task_refuses_two_embedding_sources_before_it_is_built():
    """A key that names both is invalid, not a run that fails at the first batch.

    The registry validates conditions while it expands keys, so a grid that
    varies the embedding source drops the impossible combination up front. A
    model whose encoder is pretrained counts as a source, since its tokenizer
    comes with it.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    assert ToyTaskConfig.default().validate_conditions(strict=False).passed

    pairs = [
        {"model": TRANSFORMER_FR, "embeddings": "vectors.txt"},
        {"model": TRANSFORMER_FR, "one_hot_embeddings": 8},
        {"embeddings": "vectors.txt", "one_hot_embeddings": 8},
    ]
    for pair in pairs:
        config = ToyTaskConfig.default()
        for name, value in pair.items():
            setattr(config, name, value)
        result = config.validate_conditions(strict=False)

        assert not result.passed, pair
        assert "one_embedding_source" in result.error_message


def test_a_task_configuration_declares_every_task_parameter():
    """A parameter a key cannot carry is one no registered run can record."""
    import inspect

    from pyhighlights.components.tasks import SPPTask

    parameters = set(inspect.signature(SPPTask.__init__).parameters) - {
        "self",
        "kwargs",
    }

    assert parameters - set(ToyTaskConfig.model_fields) == set()


@pytest.mark.parametrize(
    ("update", "condition"),
    [
        ({"selectors": [MLP_SELECTOR]}, "one_backbone_per_generator"),
        (
            {"selectors": [MLP_SELECTOR], "selector_backbones": [GRU_BACKBONE]},
            "at_least_two_generators",
        ),
        ({"inference_head": 3}, "inference_head_exists"),
    ],
)
def test_mgr_refuses_a_generator_set_it_cannot_run(update, condition):
    """MGR's three shape constraints are conditions, so a grid over the
    generator count never schedules a combination that cannot train.

    The registry is built first, because ``one_token_space`` reads the
    configurations the default keys name.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    config = GRUMGRConfig.default()
    assert config.validate_conditions(strict=False).passed

    for name, value in update.items():
        setattr(config, name, value)
    result = config.validate_conditions(strict=False)

    assert not result.passed
    assert condition in result.error_message


def test_every_architecture_inherits_the_shared_shape():
    """No architecture declares the shape fields for itself.

    An architecture that declared them would not receive a field added to
    the base, and would silently train its encoder at the optimizer's rate
    when ``encoder_lr`` is set. So this asserts the inheritance rather than
    the field list, which a copy would satisfy.
    """
    from pyhighlights.configurations.base import (
        PhasedSPPModelConfig,
        SPPModelConfig,
        SPPShapeConfig,
    )
    from pyhighlights.configurations.dar import GRUDARConfig
    from pyhighlights.configurations.dr import GRUDRConfig
    from pyhighlights.configurations.fr import GRUFRConfig
    from pyhighlights.configurations.genspp import GRUGenSPPConfig
    from pyhighlights.configurations.grat import GRUGRATConfig
    from pyhighlights.configurations.mcd import GRUMCDConfig
    from pyhighlights.configurations.mrd import GRUMRDConfig

    architectures = [
        GRUDARConfig,
        GRUDRConfig,
        GRUFRConfig,
        GRUGenSPPConfig,
        GRUGRATConfig,
        GRUMCDConfig,
        GRUMGRConfig,
        GRUMRDConfig,
    ]
    assert all(issubclass(config, SPPShapeConfig) for config in architectures)

    # And the criteria split is the only thing the two branches disagree on.
    assert issubclass(GRUMCDConfig, PhasedSPPModelConfig)
    assert issubclass(GRUMRDConfig, PhasedSPPModelConfig)
    assert issubclass(GRUGRATConfig, SPPModelConfig)
    assert issubclass(GRUMGRConfig, SPPModelConfig)

    # A phased model has no flat `losses` to append a criterion to, which is
    # why MCD and MRD refuse highlight supervision rather than dropping it.
    assert "losses" not in PhasedSPPModelConfig.default().values
    assert "losses" in SPPModelConfig.default().values

    # And no architecture restates a shared field at the value it already
    # inherits. A copy satisfies the inheritance above, yet a field added
    # beside it in the base would not reach the architecture holding it.
    shared = SPPShapeConfig.model_fields
    restated = [
        f"{config.__name__}.{field}"
        for config in architectures
        for field in config.__annotations__
        if field in shared
        and config.model_fields[field].default == shared[field].default
    ]
    assert not restated, restated


@pytest.mark.parametrize(
    "update",
    [
        {"temperature": 0.0},
        {"select_over": "character"},
        {"encoder_lr": 0.0},
        {"supervise_highlights": True},
    ],
)
def test_an_invalid_model_is_refused_before_a_backbone_is_built(monkeypatch, update):
    """A backbone can load a pretrained encoder, so the checks come first."""

    def refuse(*args, **kwargs):
        raise AssertionError("a backbone was built before the model was checked")

    monkeypatch.setattr(Registry, "from_keys", refuse)
    with pytest.raises(ValueError):
        FR(
            name="fr",
            losses=[],
            optimizer=ADAM,
            selector_backbones=GRU_BACKBONE,
            selectors=MLP_SELECTOR,
            predictor=MLP_PREDICTOR,
            **update,
        )


def test_the_baselines_select_as_their_originals_do():
    """FR, MCD, MGR and DAR sample at evaluation, and G-RAT thresholds at 0.5,
    its argmax. None of the five repairs an empty selection.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    for key, samples in (
        (GRU_FR, True),
        (GRU_MCD, True),
        (GRU_MGR, True),
        (GRU_GRAT, False),
        (GRU_DAR, True),
    ):
        config = Registry.retrieve_configuration(key)
        assert config.sample_at_evaluation is samples
        assert not config.repair_empty_selections


def test_every_registered_field_is_described():
    """A parameter documents itself in ``description``, not in a comment.

    A subclass that redeclares a field inherits the description, so this
    checks what a registered configuration carries rather than its source.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    undescribed = sorted(
        f"{type(config).__name__}.{name}"
        for key in Registry.retrieve_keys(namespaces=str(keys.NAMESPACE))
        for config in [Registry.retrieve_configuration(key)]
        for name, field in type(config).model_fields.items()
        if not field.description
    )
    assert not undescribed, undescribed


def test_every_key_type_names_a_class():
    """A dotted key type is text nothing resolves, so a rename leaves it stale."""
    import typing

    from cinnamon.utility.registration import import_class_from_string

    Registry.build(directory=Path(pyhighlights.__file__).parent)

    paths = {
        argument
        for key in Registry.retrieve_keys(namespaces=str(keys.NAMESPACE))
        for field in type(Registry.retrieve_configuration(key)).model_fields.values()
        for alias in [field.annotation, *typing.get_args(field.annotation)]
        for alias in [alias, *typing.get_args(alias)]
        if typing.get_origin(alias) is RegistrationKey
        for argument in typing.get_args(alias)
    }
    assert paths
    assert all(isinstance(import_class_from_string(path), type) for path in paths)


def test_the_registry_builds_without_a_component_library():
    """A build imports no torch, Lightning, torchmetrics or transformers.

    Configurations bind components by import path and name key types as
    dotted strings, so building every key costs no component import. A fresh
    interpreter, because this suite has imported all four already.
    """
    import subprocess
    import sys

    code = (
        "import sys; from pathlib import Path; import pyhighlights; "
        "from cinnamon.registry import Registry; "
        "valid, invalid = Registry.build("
        "directory=Path(pyhighlights.__file__).parent); "
        "assert valid and not invalid, invalid; "
        "print(sorted({'torch', 'lightning', 'torchmetrics', 'transformers'} "
        "& set(sys.modules)))"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


def test_a_model_reads_one_token_space():
    """A guider's backbone counts, so G-RAT's three encoders agree or fail."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert token_spaces(GRU_FR) == {None}
    assert token_spaces(GRU_GRAT) == {None}
    assert token_spaces(TRANSFORMER_FR) == {"distilbert-base-uncased"}
    assert token_spaces(TRANSFORMER_GRAT) == {"distilbert-base-uncased"}

    mixed = GRUGRATConfig.default()
    mixed.selector_backbones = TRANSFORMER_BACKBONE
    result = Registry.resolve_configuration(mixed).validate_conditions(strict=False)
    assert not result.passed
    assert "one_token_space" in result.error_message


#: A downstream project whose G-RAT varies its three encoders independently.
GRID = """
from cinnamon.configuration import Param
from cinnamon.registry import Namespace, RegistrationKey, register_class

from pyhighlights.configurations.grat import GRUGRATConfig
from pyhighlights.configurations.keys import (
    GRU_BACKBONE,
    GUIDANCE_GUIDER,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_GUIDER,
)

GRID = Namespace("grid").key("model")


@register_class(key=GRID, component="pyhighlights.components.models.spp.grat.GRAT")
class GridGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey = Param(
        GRU_BACKBONE, variants=[TRANSFORMER_BACKBONE]
    )
    predictor_backbone: RegistrationKey = Param(
        GRU_BACKBONE, variants=[TRANSFORMER_BACKBONE]
    )
    guider: RegistrationKey = Param(GUIDANCE_GUIDER, variants=[TRANSFORMER_GUIDER])
"""


def test_a_grid_over_backbones_drops_the_cells_that_disagree(tmp_path):
    """Eight cells, and only the two whose encoders read one token space stay."""
    (tmp_path / "configurations").mkdir()
    (tmp_path / "configurations" / "__init__.py").write_text("")
    (tmp_path / "configurations" / "grid.py").write_text(GRID)

    valid, invalid = Registry.build(
        directory=tmp_path,
        external_directories=[Path(pyhighlights.__file__).parent],
    )

    def cells(keys):
        return [key for key in keys if key.namespace == "grid"]

    assert len(cells(valid)) + len(cells(invalid)) == 8
    kept = [
        Registry.retrieve_variant_values(cell) for cell in sorted(cells(valid), key=str)
    ]
    assert kept == [
        {
            "selector_backbones": GRU_BACKBONE,
            "predictor_backbone": GRU_BACKBONE,
            "guider": GUIDANCE_GUIDER,
        },
        {
            "selector_backbones": TRANSFORMER_BACKBONE,
            "predictor_backbone": TRANSFORMER_BACKBONE,
            "guider": TRANSFORMER_GUIDER,
        },
    ]
    assert all("one_token_space" in str(key.metadata) for key in cells(invalid))


def test_no_registered_key_fails_its_conditions():
    """A base key that fails a condition is dropped, not reported.

    It would surface only when something asks for it, as a key that is not
    registered. A variant may fail, since pruning is what conditions are for,
    and the library registers none.
    """
    _, invalid = Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert not invalid, {str(key): key.metadata for key in invalid}


# -- what a configuration refuses before a component does -------------------


@pytest.mark.parametrize("search", [GRUGenSPPTrainerConfig, ToyGenSPPTrainerConfig])
@pytest.mark.parametrize(
    "values",
    [
        {"population_size": 51},
        {"mutation_std": math.inf},
        {"threshold_mutation_std": 0.0},
        {"threshold_mutation_std": math.nan},
        {"task_loss_limit": math.inf},
        {"stop_threshold": math.inf},
        {"devices": []},
    ],
)
def test_a_search_refuses_what_its_trainer_would(search, values):
    """Each value raises in ``GenSPPTrainer.__init__``, so the key refuses it."""
    with pytest.raises(ValidationError):
        search(**values)


def test_a_search_that_draws_no_couple_fails_its_condition():
    search = GRUGenSPPTrainerConfig.default()
    assert search.validate_conditions(strict=False).passed

    search.population_size, search.selection_rate = 4, 0.1
    result = search.validate_conditions(strict=False)
    assert not result.passed
    assert "draws_a_couple" in str(result.error_message)


def test_the_random_selector_refuses_an_unknown_mode_and_crossed_rates():
    with pytest.raises(ValidationError):
        RandomSelectorConfig(mode="words")

    selector = RandomSelectorConfig.default()
    selector.min_keep, selector.max_keep = 0.8, 0.2
    result = selector.validate_conditions(strict=False)
    assert not result.passed
    assert "ordered_keep_rates" in str(result.error_message)


def test_the_reported_class_is_a_class():
    with pytest.raises(ValidationError):
        HighlightMetricConfig(pos_label=-1)

    metric = ClassF1Config.default()
    metric.pos_label = 2
    result = metric.validate_conditions(strict=False)
    assert not result.passed
    assert "pos_label_is_a_class" in str(result.error_message)


@pytest.mark.parametrize(
    "configuration, values",
    [
        (ToyTaskConfig, {"vocabulary_from": "words"}),
        (GRUGRATConfig, {"select_over": "characters"}),
    ],
)
def test_a_choice_refuses_a_value_its_component_does_not_know(configuration, values):
    with pytest.raises(ValidationError):
        configuration(**values)


@pytest.mark.parametrize(
    "component, configuration",
    [
        (
            "pyhighlights.components.models.spp.genspp.GenSPPTrainer",
            GRUGenSPPTrainerConfig,
        ),
        ("pyhighlights.utility.losses.CrossEntropy", CrossEntropyConfig),
        ("pyhighlights.utility.losses.MaskedCrossEntropy", MaskedCrossEntropyConfig),
    ],
)
def test_a_configuration_declares_every_parameter_of_its_component(
    component, configuration
):
    from cinnamon.utility.registration import import_class_from_string

    parameters = inspect.signature(import_class_from_string(component)).parameters
    assert set(parameters) <= set(configuration.model_fields)


def test_a_task_leaves_the_device_to_lightning():
    """Only a task that must stay on the CPU, such as a toy one, pins it."""
    assert "accelerator" not in TaskConfig.model_fields["trainer_args"].default
    assert ToyTaskConfig().trainer_args["accelerator"] == "cpu"


def test_an_override_keeps_the_constraints_of_its_base():
    """Pydantic replaces a redeclared field whole, constraints included.

    A subclass that changes only a default would otherwise accept what its
    base refuses, as an override of ``population_size`` once accepted an odd
    population. ``RemainingDiscrepancyLossConfig.coefficient`` is negative on
    purpose: it is the one term a model maximizes.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    exempt = {("RemainingDiscrepancyLossConfig", "coefficient")}

    dropped = set()
    for key in Registry.retrieve_keys(namespaces=str(keys.NAMESPACE)):
        for klass in type(Registry.retrieve_configuration(key)).__mro__:
            if not (isinstance(klass, type) and issubclass(klass, BaseModel)):
                continue
            for name in klass.__dict__.get("__annotations__", {}):
                base = next(
                    (
                        parent.model_fields[name]
                        for parent in klass.__mro__[1:]
                        if issubclass(parent, BaseModel) and name in parent.model_fields
                    ),
                    None,
                )
                own = klass.model_fields.get(name)
                if base is None or own is None:
                    continue
                if set(map(repr, base.metadata)) - set(map(repr, own.metadata)):
                    dropped.add((klass.__name__, name))
    assert dropped == exempt
