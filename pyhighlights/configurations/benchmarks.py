"""Benchmark and analyzer registrations."""

from typing import Any, Dict, List, Sequence, Set

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import (
    HIGHLIGHT_POSITION_ANALYZER,
    LABEL_STUDIO_EXPORTER,
    METRICS_ANALYZER,
    PREDICTION_ANALYZER,
    TOY_BENCHMARK,
)
from pyhighlights.constants import PREDICTIONS


class BenchmarkConfig(Configuration):
    """The tasks a registry search selects, run in key order.

    Each filter left as ``None`` matches every key. ``task_tags`` matches keys
    holding every listed tag, and ``task_variants`` matches values of variant
    parameters, defaults included, with dotted names reading into child keys.
    """

    task_name: str = Param(
        "task", description="Registration name that a selected task key must have."
    )
    task_namespace: str | None = Param(
        None,
        description=(
            "Namespace that a selected task key must belong to, or ``None`` to "
            "match every namespace."
        ),
    )
    task_tags: Set[str] | None = Param(
        None,
        description=(
            "Tags that a selected task key must all hold, or ``None`` to match "
            "every key."
        ),
    )
    task_variants: Dict[str, Any] | None = Param(
        None,
        description=(
            "Variant parameter values, defaults included, that a selected task "
            "key must have, with dotted names reading into child keys. "
            "``None`` matches every key."
        ),
    )
    name: str = Param(
        "benchmark",
        description=(
            "Name of the benchmark, used in its report and as the directory "
            "under ``save_path`` that its tasks write to."
        ),
    )
    save_path: str | None = Param(
        None,
        description=(
            "Directory that holds the benchmark directory, or ``None`` for ``results``."
        ),
    )
    strict: bool = Param(
        False,
        description=(
            "Whether a failing task stops the benchmark instead of being recorded "
            "as failed while the remaining tasks run."
        ),
    )
    task_args: Dict[str, Any] = Param(
        {},
        description=(
            "Keyword arguments passed to every task this benchmark builds, so a "
            "registered grid can run differently, for example with one seed or a "
            "smaller batch, without registering a second one. Each task's "
            "manifest records what it was built with, so an overridden run says so."
        ),
    )


@register_class(
    key=TOY_BENCHMARK,
    component="pyhighlights.components.benchmarks.Benchmark",
    run_method="run",
)
class ToyBenchmarkConfig(BenchmarkConfig):
    """The synthetic corpus, as a benchmark of one: the end-to-end sanity check run."""

    task_namespace: str | None = Param("pyhighlights")
    task_tags: Set[str] | None = Param({"spp", "toy"})
    name: str = Param("toy-benchmark")


@register_class(
    key=METRICS_ANALYZER,
    component="pyhighlights.components.analyzers.MetricsAnalyzer",
    run_method="run",
)
class MetricsAnalyzerConfig(Configuration):
    """One row per task, ``mean +/- std`` across seeds."""

    directory: str | None = Param(
        None,
        description=(
            "Results directory searched recursively for runs, or ``None`` for "
            "``results``."
        ),
    )
    metrics: Sequence[str] = Param(
        [],
        description=(
            "Metric names to report, as columns in this order, or empty to report "
            "every metric found."
        ),
    )
    split: str = Param("test", description="Split whose metrics the table reports.")
    pairs: bool = Param(
        False,
        description=(
            "Whether each cell holds a ``(mean, std)`` pair instead of the text "
            "``mean +/- std``."
        ),
    )
    latest: bool = Param(
        True,
        description=(
            "Whether to report only the most recent run of each task instead of "
            "every run a task has ever done."
        ),
    )


@register_class(
    key=HIGHLIGHT_POSITION_ANALYZER,
    component="pyhighlights.components.analyzers.HighlightPositionAnalyzer",
    run_method="run",
)
class HighlightPositionAnalyzerConfig(Configuration):
    """Where in the document the selector looked."""

    directory: str | None = Param(
        None,
        description=(
            "Results directory searched recursively for runs, or ``None`` for "
            "``results``."
        ),
    )
    pattern: str = Param(
        PREDICTIONS, description="File name pattern of the stored predictions to read."
    )
    bins: int = Param(
        10,
        ge=1,
        description=(
            "Number of position columns, at least 1: shares of the document, or "
            "the first ``bins`` word positions when ``absolute`` is set."
        ),
    )
    absolute: bool = Param(
        False,
        description=(
            "Whether to report word positions instead of positions as a share of "
            "the document length."
        ),
    )
    latest: bool = Param(
        True,
        description=(
            "Whether to read only the newest run of each task, as every other "
            "analyzer does, so a re-run task is one block of rows here and one "
            "row in the metrics table."
        ),
    )


@register_class(
    key=PREDICTION_ANALYZER,
    component="pyhighlights.components.analyzers.PredictionAnalyzer",
    run_method="run",
)
class PredictionAnalyzerConfig(Configuration):
    """What the selector kept, in words, one row per sample."""

    directory: str | None = Param(
        None,
        description=(
            "Results directory searched recursively for runs, or ``None`` for "
            "``results``."
        ),
    )
    pattern: str = Param(
        PREDICTIONS, description="File name pattern of the stored predictions to read."
    )
    split: str = Param(
        "test", description="Split of the run's corpus that the predictions cover."
    )
    latest: bool = Param(
        True,
        description=(
            "Whether to read only the newest run of each task, as the metrics "
            "analyzer does. ``False`` reads every run a task has ever done, which "
            "reports on the history rather than on the model."
        ),
    )


@register_class(
    key=LABEL_STUDIO_EXPORTER,
    component="pyhighlights.components.analyzers.LabelStudioExporter",
    run_method="run",
)
class LabelStudioExporterConfig(PredictionAnalyzerConfig):
    """Predicted highlights, written where a domain expert can read them."""

    model_version: str = Param(
        "pyhighlights",
        description="Model version written into each Label Studio pre-annotation.",
    )
    labels: Sequence[str] = Param(
        ["highlight"],
        description="Label Studio labels given to each highlighted word span.",
    )
    only: int | None = Param(
        None,
        description=(
            "Class that an exported sample must have in ``column``, or ``None`` "
            "to export every sample."
        ),
    )
    column: str = Param(
        "label",
        description=(
            "Prediction column that ``only`` is matched against: ``label`` for the "
            "annotated class or ``predicted`` for the predicted class."
        ),
    )
    stem: str = Param(
        "label-studio",
        description=(
            "File name prefix of each export, written beside its predictions as "
            "``<stem>-seed=<seed>.json``."
        ),
    )


__all__: List[str] = [
    "BenchmarkConfig",
    "LabelStudioExporterConfig",
    "HighlightPositionAnalyzerConfig",
    "MetricsAnalyzerConfig",
    "PredictionAnalyzerConfig",
    "ToyBenchmarkConfig",
]
