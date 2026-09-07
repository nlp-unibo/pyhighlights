from typing import List

import pytest
import torch as th
import torchmetrics
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.metrics import (
    BinaryHighlightF1Score,
    BinaryHighlightIoU,
    SelectionRate,
    SelectionSize,
)
from pyhighlights.utility.metrics import BoundMetric, build_metrics


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
        component="pyhighlights.metrics.BinaryHighlightF1Score",
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
