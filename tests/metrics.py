import pytest
import torchmetrics

from pyhighlights.utility.metrics import build_torchmetric, build_torchmetrics

# TODO: update
def test_creating_metric_flat():
    metric = build_torchmetric(name="F1Score", task="multiclass", num_classes=5)
    assert isinstance(metric, torchmetrics.Metric)
    assert isinstance(metric, torchmetrics.classification.MulticlassF1Score)


def test_error_creating_metric_flat():
    with pytest.raises(ValueError):
        build_torchmetric("F1Scor", task="multiclass", num_classes=5)


def test_creating_metric_path():
    metric = build_torchmetric(
        name="classification.F1Score", task="multiclass", num_classes=5
    )
    assert isinstance(metric, torchmetrics.Metric)
    assert isinstance(metric, torchmetrics.classification.MulticlassF1Score)


def test_error_creating_metric_path():
    with pytest.raises(ValueError):
        build_torchmetric("classification.F1Scor", task="multiclass", num_classes=5)


def test_error_creating_metric_path_submodule():
    with pytest.raises(ValueError):
        build_torchmetric(
            "not_a_real_submodule.F1Score", task="multiclass", num_classes=5
        )


def test_creating_metrics():
    metrics = build_torchmetrics(
        {
            "f1": {"name": "F1Score", "task": "multiclass", "num_classes": 5},
            "acc": {"name": "Accuracy", "task": "multiclass", "num_classes": 5},
        }
    )
    assert isinstance(metrics, torchmetrics.MetricCollection)
    assert isinstance(metrics["f1"], torchmetrics.classification.MulticlassF1Score)
    assert isinstance(metrics["acc"], torchmetrics.classification.MulticlassAccuracy)
