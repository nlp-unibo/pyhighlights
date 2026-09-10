"""Benchmark and analyzer registrations."""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.analyzers import PREDICTIONS
from pyhighlights.components.tasks import Task
from pyhighlights.configurations.keys import NAMESPACE, TOY_TASK


class BenchmarkConfig(Configuration):
    """A grid of tasks, run in order."""

    tasks: List[RegistrationKey[Task]] = Param([TOY_TASK])
    name: str = Param("benchmark")
    save_path: str | None = Param(None)
    strict: bool = Param(False)
    #: Passed to every task this benchmark builds, so a registered grid can be
    #: run differently without registering a second one -- one batch and one
    #: seed to check that every cell holds together, or a smaller batch for a
    #: card that cannot fit the registered one. Each task's manifest records
    #: what it was built with, so an overridden run says so.
    task_args: Dict[str, Any] = Param({})


@register_class(
    name="benchmark",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.benchmarks.Benchmark",
    run_method="run",
)
class ToyBenchmarkConfig(BenchmarkConfig):
    """The synthetic corpus, as a benchmark of one: the end-to-end smoke test."""

    name: str = Param("toy-benchmark")


@register_class(
    name="analyzer",
    tags={"metrics"},
    namespace=NAMESPACE,
    component="pyhighlights.components.analyzers.MetricsAnalyzer",
    run_method="run",
)
class MetricsAnalyzerConfig(Configuration):
    """One row per task, ``mean +/- std`` across seeds."""

    directory: str | None = Param(None)
    metrics: Sequence[str] = Param([])
    split: str = Param("test")
    pairs: bool = Param(False)
    #: The most recent run of each task, or every run a task has ever done.
    latest: bool = Param(True)


@register_class(
    name="analyzer",
    tags={"highlight", "position"},
    namespace=NAMESPACE,
    component="pyhighlights.components.analyzers.HighlightPositionAnalyzer",
    run_method="run",
)
class HighlightPositionAnalyzerConfig(Configuration):
    """Where in the document the selector looked."""

    directory: str | None = Param(None)
    pattern: str = Param(PREDICTIONS)
    bins: int = Param(10, ge=1)
    absolute: bool = Param(False)


@register_class(
    name="analyzer",
    tags={"prediction"},
    namespace=NAMESPACE,
    component="pyhighlights.components.analyzers.PredictionAnalyzer",
    run_method="run",
)
class PredictionAnalyzerConfig(Configuration):
    """What the selector kept, in words, one row per sample."""

    directory: str | None = Param(None)
    pattern: str = Param(PREDICTIONS)
    split: str = Param("test")


@register_class(
    name="analyzer",
    tags={"label-studio"},
    namespace=NAMESPACE,
    component="pyhighlights.components.analyzers.LabelStudioExporter",
    run_method="run",
)
class LabelStudioExporterConfig(PredictionAnalyzerConfig):
    """Predicted highlights, written where a domain expert can read them."""

    model_version: str = Param("pyhighlights")
    labels: Sequence[str] = Param(["highlight"])
    only: int | None = Param(None)
    #: Which class ``only`` names: the annotated one, or the predicted one.
    column: str = Param("label")
    stem: str = Param("label-studio")


__all__: List[str] = [
    "BenchmarkConfig",
    "LabelStudioExporterConfig",
    "HighlightPositionAnalyzerConfig",
    "MetricsAnalyzerConfig",
    "PredictionAnalyzerConfig",
    "ToyBenchmarkConfig",
]
