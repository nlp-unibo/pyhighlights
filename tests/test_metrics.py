import torchmetrics
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import Registry

from pyhighlights.utility.metrics import build_torchmetric, build_torchmetrics


class MulticlassMetricConfig(Configuration):
    num_classes: int = Param(3)


def test_build_registered_torchmetrics():
    Registry.initialize()
    f1 = Registry.register_configuration(
        config=MulticlassMetricConfig.default(),
        name="f1",
        namespace="tests",
        component="torchmetrics.classification.MulticlassF1Score",
    )
    accuracy = Registry.register_configuration(
        config=MulticlassMetricConfig.default(),
        name="accuracy",
        namespace="tests",
        component="torchmetrics.classification.MulticlassAccuracy",
    )
    Registry.dag_resolution()

    assert isinstance(build_torchmetric(f1), torchmetrics.Metric)
    metrics = build_torchmetrics({"f1": f1, "accuracy": accuracy})
    assert isinstance(metrics, torchmetrics.MetricCollection)
    assert isinstance(metrics["f1"], torchmetrics.classification.MulticlassF1Score)
    assert isinstance(
        metrics["accuracy"], torchmetrics.classification.MulticlassAccuracy
    )
