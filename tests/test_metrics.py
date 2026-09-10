from pathlib import Path
from typing import List

import pytest
import torch as th
import torchmetrics
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, Registry

import pyhighlights
from pyhighlights.configurations.keys import (
    ACCURACY_METRIC,
    CLASS_F1_METRIC,
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    MULTICLASS_ACCURACY_METRIC,
    MULTICLASS_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
)
from pyhighlights.utility.metrics import (
    BinaryHighlightF1Score,
    BinaryHighlightIoU,
    BoundMetric,
    ClassF1Score,
    SelectionRate,
    SelectionSize,
    build_metrics,
)


class MulticlassMetricConfig(Configuration):
    num_classes: int = Param(3)


class BoundMetricConfig(Configuration):
    name: str = Param("accuracy")
    metric: RegistrationKey = Param(
        RegistrationKey(name="class_binding", namespace="tests")
    )
    inputs: List[str] = Param(["class_logits", "y_true"])


def test_build_bound_metrics_from_registered_metrics():
    Registry.initialize()
    highlight_f1 = Registry.register_configuration(
        config=Configuration.default(),
        name="highlight_f1",
        namespace="tests",
        component="pyhighlights.utility.metrics.BinaryHighlightF1Score",
    )
    class_binding = Registry.register_configuration(
        config=MulticlassMetricConfig.default(),
        name="class_binding",
        namespace="tests",
        component="torchmetrics.classification.MulticlassAccuracy",
    )
    binding = Registry.register_configuration(
        config=BoundMetricConfig.default(),
        name="binding",
        namespace="tests",
        component="pyhighlights.utility.metrics.BoundMetric",
    )
    Registry.dag_resolution()

    accuracy = BoundMetric(name="accuracy", metric=class_binding)
    highlight = BoundMetric(
        name="highlight_f1",
        metric=highlight_f1,
        inputs=["highlight_mask", "highlight_true"],
    )
    assert isinstance(accuracy.metric, torchmetrics.Metric)
    assert accuracy.inputs == ["class_logits", "y_true"]

    values = {
        "class_logits": th.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "y_true": th.tensor([0, 1]),
        "highlight_mask": th.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "highlight_true": th.tensor([[1, 0], [1, -1]]),
    }
    accuracy.update(values)
    highlight.update(values)

    assert accuracy.compute() == 1.0
    assert highlight.compute() == 1.0

    highlight.reset()
    with pytest.raises(KeyError, match="highlight_true"):
        highlight.update({"highlight_mask": values["highlight_mask"]})

    built = build_metrics([binding])
    assert isinstance(built, th.nn.ModuleList)
    assert [metric.name for metric in built] == ["accuracy"]
    assert len(build_metrics(None)) == 0


def test_highlight_metrics_ignore_unlabelled_positions():
    preds = th.tensor([[1.0, 1.0, 0.0]])
    target = th.tensor([[1, 0, -1]])

    f1 = BinaryHighlightF1Score()
    f1.update(preds, target)
    iou = BinaryHighlightIoU()
    iou.update(preds, target)

    assert f1.compute() == pytest.approx(2 / 3)
    assert iou.compute() == pytest.approx(0.5)


def test_selection_metrics_average_over_samples():
    preds = th.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    target = th.tensor([[1, 0, -1], [1, 0, 0]])

    rate = SelectionRate()
    rate.update(preds, target)
    size = SelectionSize()
    size.update(preds, target)

    assert rate.compute() == pytest.approx((1.0 + 1 / 3) / 2)
    assert size.compute() == pytest.approx(1.5)


def test_registered_metrics_score_the_fields_they_name():
    """The registered set is what a task hands a model, so it must bind."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        # Two classes, one logit each: the classification metrics are
        # registered as multiclass even for a binary corpus.
        "class_logits": th.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "y_true": th.tensor([0, 1]),
        "highlight_mask": th.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "highlight_true": th.tensor([[1, 0], [1, -1]]),
        "mask": th.tensor([[1.0, 1.0], [1.0, 1.0]]),
    }
    scores = {}
    for key in (
        ACCURACY_METRIC,
        F1_METRIC,
        HIGHLIGHT_F1_METRIC,
        HIGHLIGHT_IOU_METRIC,
        SELECTION_RATE_METRIC,
        SELECTION_SIZE_METRIC,
    ):
        metric = Registry.from_key(key, expected_type=BoundMetric)
        metric.update(values)
        scores[metric.name] = metric.compute().item()

    assert scores["accuracy"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(1.0)
    # Two annotated positions are marked and both are selected; the third is
    # unannotated and scores nothing.
    assert scores["highlight_f1"] == pytest.approx(1.0)
    assert scores["highlight_iou"] == pytest.approx(1.0)
    assert scores["selection_rate"] == pytest.approx(0.75)
    assert scores["selection_size"] == pytest.approx(1.5)


def test_multiclass_metrics_are_registered_for_hatexplain():
    """HateXplain has three classes; the binary registrations cannot score it."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        "class_logits": th.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        "y_true": th.tensor([0, 1]),
    }
    accuracy = Registry.from_key(MULTICLASS_ACCURACY_METRIC, expected_type=BoundMetric)
    accuracy.update(values)

    assert accuracy.compute().item() == pytest.approx(0.5)

    f1 = Registry.from_key(MULTICLASS_F1_METRIC, expected_type=BoundMetric)
    f1.update(values)
    assert 0.0 <= f1.compute().item() <= 1.0


def test_each_build_gets_its_own_metric_state():
    """Two models must not share a metric: seeds would accumulate each other."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    first = Registry.from_key(ACCURACY_METRIC, expected_type=BoundMetric)
    second = Registry.from_key(ACCURACY_METRIC, expected_type=BoundMetric)
    first.update({"class_logits": th.tensor([[2.0, 0.0]]), "y_true": th.tensor([0])})

    assert first.metric is not second.metric
    assert second.metric.compute().item() == pytest.approx(0.0)


def test_class_f1_reports_the_rare_class_where_macro_reports_the_other_one():
    """Nineteen negatives and one positive, all called negative.

    Macro F1 answers 0.49 -- half of a perfect score on the class that is 95%
    of the rows -- and the model found nothing. The class metric answers 0.0,
    which is the number a skewed corpus is read with.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        "class_logits": th.tensor([[2.0, 0.0]] * 20),
        "y_true": th.tensor([0] * 19 + [1]),
    }

    macro = Registry.from_key(F1_METRIC, expected_type=BoundMetric)
    macro.update(values)
    assert macro.compute().item() == pytest.approx(0.4872, abs=1e-4)

    unfair = Registry.from_key(CLASS_F1_METRIC, expected_type=BoundMetric)
    unfair.update(values)
    # Reported under the same name: which F1 it is, the manifest's key says.
    assert unfair.name == "f1"
    assert unfair.compute().item() == pytest.approx(0.0)

    found = Registry.from_key(CLASS_F1_METRIC, expected_type=BoundMetric)
    found.update(
        {
            "class_logits": th.tensor([[2.0, 0.0]] * 19 + [[0.0, 2.0]]),
            "y_true": th.tensor([0] * 19 + [1]),
        }
    )
    assert found.compute().item() == pytest.approx(1.0)


def test_class_f1_refuses_a_class_it_cannot_score():
    with pytest.raises(ValueError, match="pos_label"):
        ClassF1Score(pos_label=2, num_classes=2)
