"""Benchmark and analyzer registrations."""

from typing import List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.tasks import Task
from pyhighlights.configurations.keys import NAMESPACE, TOY_TASK


class BenchmarkConfig(Configuration):
    """A grid of tasks, run in order."""

    tasks: List[RegistrationKey[Task]] = Param([TOY_TASK])
    name: str = Param("benchmark")
    save_path: str | None = Param(None)
    strict: bool = Param(False)


class ToyBenchmarkConfig(BenchmarkConfig):
    """The synthetic corpus, as a benchmark of one: the end-to-end smoke test."""

    name: str = Param("toy-benchmark")

    @classmethod
    @register_method(
        name="benchmark",
        tags={"toy"},
        namespace=NAMESPACE,
        component="pyhighlights.components.benchmarks.Benchmark",
        run_method="run",
    )
    def default(cls):
        return super().default()


class MetricsAnalyzerConfig(Configuration):
    """One row per task, ``mean +/- std`` across seeds."""

    directory: str | None = Param(None)
    metrics: Sequence[str] = Param([])
    split: str = Param("test")
    pairs: bool = Param(False)

    @classmethod
    @register_method(
        name="analyzer",
        tags={"metrics"},
        namespace=NAMESPACE,
        component="pyhighlights.components.analyzers.MetricsAnalyzer",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HighlightPositionAnalyzerConfig(Configuration):
    """Where in the document the selector looked."""

    directory: str | None = Param(None)
    filename: str = Param("predictions.pkl")
    bins: int = Param(10, ge=1)

    @classmethod
    @register_method(
        name="analyzer",
        tags={"position", "highlight"},
        namespace=NAMESPACE,
        component="pyhighlights.components.analyzers.HighlightPositionAnalyzer",
        run_method="run",
    )
    def default(cls):
        return super().default()


__all__: List[str] = [
    "BenchmarkConfig",
    "HighlightPositionAnalyzerConfig",
    "MetricsAnalyzerConfig",
    "ToyBenchmarkConfig",
]
