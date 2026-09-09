from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import RegistrationKey, Registry

import pyhighlights
from pyhighlights.components import faithfulness
from pyhighlights.components.models import InputData
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp import FR
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import TOY_TASK
from tests.test_fr import NAMESPACE, register


def tiny_model() -> FR:
    Registry.initialize()
    backbone = register("backbone", "tests.test_fr.TinyBackbone")
    selector = register("selector", "tests.test_fr.TinySelector")
    predictor = register("predictor", "tests.test_fr.TinyPredictor")
    Registry.dag_resolution()

    model = FR(
        name="fr",
        losses=[],
        optimizer=RegistrationKey(name="optimizer", namespace=NAMESPACE),
        selector_backbones=backbone,
        selectors=selector,
        predictor=predictor,
    )
    model.eval()
    return model


def batch() -> InputData:
    return InputData(
        features=th.tensor([[1, 2, 3], [4, 5, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 3), -1),
    )


def output_for(model: FR, data: InputData, highlight: th.Tensor) -> SPPOutput:
    """An output whose highlight is dictated rather than selected.

    ``class_logits`` are the full-input pass, which pins the class ``y_hat``
    and makes both identities below exact rather than approximate.
    """
    return SPPOutput(
        class_logits=model.predict_full(data).unsqueeze(1),
        highlight_logits=th.zeros((*highlight.shape, 2)).unsqueeze(1),
        highlight_mask=highlight.unsqueeze(1),
    )


def test_a_highlight_covering_everything_is_exactly_sufficient():
    model, data = tiny_model(), batch()

    with th.no_grad():
        terms = model.faithfulness(data, output_for(model, data, data.mask))

    # The highlight is the whole input, so p(y_hat | h) is p(y_hat | x).
    assert th.allclose(terms["sufficiency"], th.zeros(2), atol=1e-6)
    # Its complement is empty, so comprehensiveness is the whole confidence
    # the predictor has left when it reads nothing.
    assert (terms["comprehensiveness"].abs() <= 1.0).all()


def test_an_empty_highlight_is_exactly_uncomprehensive():
    model, data = tiny_model(), batch()

    with th.no_grad():
        terms = model.faithfulness(
            data, output_for(model, data, th.zeros_like(data.mask))
        )

    # Nothing was removed, so the complement pass is the full-input pass.
    assert th.allclose(terms["comprehensiveness"], th.zeros(2), atol=1e-6)


def test_a_model_that_does_not_measure_faithfulness_says_so():
    model, data = tiny_model(), batch()
    output = output_for(model, data, data.mask)

    # Reached through the base implementation, which is what a model outside
    # the select-then-predict family inherits.
    with pytest.raises(NotImplementedError, match="does not measure faithfulness"):
        Model.faithfulness(model, data, output)


def test_evaluate_averages_over_the_split_and_restores_the_mode():
    model, data = tiny_model(), batch()
    model.train()

    terms = faithfulness.evaluate(model, [data, data])

    assert set(terms) == {"sufficiency", "comprehensiveness"}
    assert all(-1.0 <= value <= 1.0 for value in terms.values())
    # Scoring a split is not training on it, and the caller's mode survives.
    assert model.training


def test_evaluate_refuses_an_empty_split():
    model = tiny_model()

    with pytest.raises(ValueError, match="at least one example"):
        faithfulness.evaluate(model, [])


def test_a_task_reports_faithfulness_only_when_asked(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        faithfulness=True,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    results = task.run()

    assert isinstance(task, SPPTask)
    run = results["runs"][0]
    assert "test_sufficiency" in run and "test_comprehensiveness" in run
    assert -1.0 <= run["test_comprehensiveness"] <= 1.0

    plain = Registry.from_key(
        TOY_TASK,
        name="toy-plain",
        save_path=str(tmp_path),
        seeds=[0],
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    assert "test_sufficiency" not in plain.run()["runs"][0]
