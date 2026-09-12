"""Metric registrations, and the bindings that feed them named fields.

Two layers, as with the losses: a ``torchmetric`` is the scoring object, and a
``metric`` binds it to the fields of the step namespace it reads. The binding
is what a model or a task names, since the same F1 scores classes or tokens
depending on what it is handed.

Classification metrics are registered per class count, because
``torchmetrics`` needs to know: Beer, Hotel, Movies and Toy have two classes,
HateXplain has three. Both use ``task="multiclass"`` -- a predictor emits one
logit per class, including when there are two, and the ``"binary"`` task wants
a single score per sample instead. Highlight and selection metrics are
class-agnostic -- a token is selected or it is not -- so they are registered
once and used by every corpus.
"""

from typing import List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class
from torchmetrics import Metric

from pyhighlights.configurations.keys import (
    ACCURACY,
    CLASS_F1,
    EMPTY_SET_ACCURACY,
    EXACT_SET_MATCH,
    F1,
    HIGHLIGHT_F1,
    HIGHLIGHT_IOU,
    LINK_F1,
    LINK_MACRO_F1,
    MULTICLASS_ACCURACY,
    MULTICLASS_F1,
    NAMESPACE,
    SELECTION_RATE,
    SELECTION_SIZE,
)

#: Classes the multiclass variants score. HateXplain is the corpus that needs
#: them: ``hatespeech``, ``normal`` and ``offensive``.
MULTICLASS_CLASSES = 3


BOUND_METRIC_COMPONENT = "pyhighlights.utility.metrics.BoundMetric"
ACCURACY_COMPONENT = "torchmetrics.Accuracy"
F1_SCORE_COMPONENT = "torchmetrics.F1Score"


@register_class(
    name="torchmetric",
    tags={"accuracy"},
    namespace=NAMESPACE,
    component=ACCURACY_COMPONENT,
)
class AccuracyConfig(Configuration):
    task: str = Param("multiclass")
    num_classes: int = Param(2, ge=2)
    average: str = Param("micro")


@register_class(
    name="torchmetric",
    tags={"accuracy", "multiclass"},
    namespace=NAMESPACE,
    component=ACCURACY_COMPONENT,
)
class MulticlassAccuracyConfig(AccuracyConfig):
    num_classes: int = Param(MULTICLASS_CLASSES, ge=2)


@register_class(
    name="torchmetric", tags={"f1"}, namespace=NAMESPACE, component=F1_SCORE_COMPONENT
)
class F1Config(Configuration):
    task: str = Param("multiclass")
    num_classes: int = Param(2, ge=2)
    average: str = Param("macro")


@register_class(
    name="torchmetric",
    tags={"f1", "multiclass"},
    namespace=NAMESPACE,
    component=F1_SCORE_COMPONENT,
)
class MulticlassF1Config(F1Config):
    num_classes: int = Param(MULTICLASS_CLASSES, ge=2)


@register_class(
    name="torchmetric",
    tags={"f1", "class"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.ClassF1Score",
)
class ClassF1Config(Configuration):
    """F1 of one class, for a corpus an average would flatter.

    ``pos_label`` is the class the run is about -- the rare one, on a corpus
    skewed enough for macro F1 to be mostly the majority class.
    """

    pos_label: int = Param(1, ge=0)
    num_classes: int = Param(2, ge=2)


class HighlightMetricConfig(Configuration):
    """Token-level scores over the positions a corpus annotated.

    ``ignore_index`` marks a position that carries no annotation, which is what
    an unannotated split is padded with; those positions score nothing rather
    than counting as negatives.
    """

    pos_label: int = Param(1)
    ignore_index: int = Param(-1)


@register_class(
    name="torchmetric",
    tags={"f1", "highlight"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.BinaryHighlightF1Score",
)
class HighlightF1Config(HighlightMetricConfig):
    pass


@register_class(
    name="torchmetric",
    tags={"highlight", "iou"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.BinaryHighlightIoU",
)
class HighlightIoUConfig(HighlightMetricConfig):
    pass


@register_class(
    name="torchmetric",
    tags={"selection_rate"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.SelectionRate",
)
class SelectionRateConfig(Configuration):
    """Share of a document the selector kept, averaged over samples.

    No ``ignore_index``: the metric is bound to ``mask``, so what it has to
    exclude is padding rather than an unannotated position, and a mask says
    that with a zero.
    """


@register_class(
    name="torchmetric",
    tags={"selection_size"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.SelectionSize",
)
class SelectionSizeConfig(SelectionRateConfig):
    """Tokens the selector kept, averaged over samples."""


@register_class(
    name="metric",
    tags={"accuracy"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class MetricConfig(Configuration):
    """A scoring object bound to the namespace fields it reads."""

    name: str = Param("accuracy")
    metric: RegistrationKey[Metric] = Param(ACCURACY)
    inputs: Sequence[str] = Param(["class_logits", "y_true"])


@register_class(
    name="metric",
    tags={"accuracy", "multiclass"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class MulticlassAccuracyMetricConfig(MetricConfig):
    metric: RegistrationKey[Metric] = Param(MULTICLASS_ACCURACY)


@register_class(
    name="metric", tags={"f1"}, namespace=NAMESPACE, component=BOUND_METRIC_COMPONENT
)
class F1MetricConfig(MetricConfig):
    name: str = Param("f1")
    metric: RegistrationKey[Metric] = Param(F1)


@register_class(
    name="metric",
    tags={"f1", "multiclass"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class MulticlassF1MetricConfig(F1MetricConfig):
    metric: RegistrationKey[Metric] = Param(MULTICLASS_F1)


@register_class(
    name="metric",
    tags={"f1", "class"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class ClassF1MetricConfig(F1MetricConfig):
    """Still reported as ``f1``: which F1 it is, the manifest's key says."""

    metric: RegistrationKey[Metric] = Param(CLASS_F1)


@register_class(
    name="metric",
    tags={"f1", "highlight"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class HighlightF1MetricConfig(MetricConfig):
    """Scores the selected tokens against the annotated ones."""

    name: str = Param("highlight_f1")
    metric: RegistrationKey[Metric] = Param(HIGHLIGHT_F1)
    inputs: Sequence[str] = Param(["highlight_mask", "highlight_true"])


@register_class(
    name="metric",
    tags={"highlight", "iou"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class HighlightIoUMetricConfig(HighlightF1MetricConfig):
    name: str = Param("highlight_iou")
    metric: RegistrationKey[Metric] = Param(HIGHLIGHT_IOU)


@register_class(
    name="metric",
    tags={"selection_rate"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class SelectionRateMetricConfig(HighlightF1MetricConfig):
    """Reports what the selector kept, whether or not the corpus is annotated.

    ``mask`` and not ``highlight_true``, so the denominator is the document's
    own length: the statistic is about the selection, and a corpus with no
    annotation still has one.
    """

    name: str = Param("selection_rate")
    metric: RegistrationKey[Metric] = Param(SELECTION_RATE)
    inputs: Sequence[str] = Param(["highlight_mask", "mask"])


@register_class(
    name="metric",
    tags={"selection_size"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class SelectionSizeMetricConfig(SelectionRateMetricConfig):
    name: str = Param("selection_size")
    metric: RegistrationKey[Metric] = Param(SELECTION_SIZE)


__all__: List[str] = [
    "AccuracyConfig",
    "ClassF1Config",
    "ClassF1MetricConfig",
    "F1Config",
    "F1MetricConfig",
    "HighlightF1Config",
    "HighlightF1MetricConfig",
    "HighlightIoUConfig",
    "HighlightIoUMetricConfig",
    "HighlightMetricConfig",
    "MetricConfig",
    "MulticlassAccuracyConfig",
    "MulticlassAccuracyMetricConfig",
    "MulticlassF1Config",
    "MulticlassF1MetricConfig",
    "SelectionRateConfig",
    "SelectionRateMetricConfig",
    "SelectionSizeConfig",
    "SelectionSizeMetricConfig",
]


#: Entries of the knowledge base a link metric scores.
#:
#: Registered per size, for the reason the classification metrics are
#: registered per class count: ``torchmetrics`` needs the number up front, and
#: a metric sized for the wrong base scores the wrong columns. ToS-100 runs
#: from 7 rationales on one category to 28 on another, so a study registers a
#: variant of these carrying its own ``num_labels`` rather than overriding one
#: at build time -- an override would not reach the metric, which a binding
#: builds from its key alone.
KNOWLEDGE_ENTRIES = 2


@register_class(
    name="torchmetric",
    tags={"f1", "link"},
    namespace=NAMESPACE,
    component="torchmetrics.classification.MultilabelF1Score",
)
class LinkF1Config(Configuration):
    """Per-link F1, micro-averaged: every (example, entry) link counts once.

    The comparable headline number. Recall is the one to watch inside it: an
    example can instantiate a dozen entries, so a model that names one correct
    entry and stops looks precise and has missed the case.
    """

    num_labels: int = Param(KNOWLEDGE_ENTRIES, ge=1)
    average: str = Param("micro")
    ignore_index: int = Param(-1)


@register_class(
    name="torchmetric",
    tags={"f1", "link", "macro"},
    namespace=NAMESPACE,
    component="torchmetrics.classification.MultilabelF1Score",
)
class LinkMacroF1Config(LinkF1Config):
    """Per-link F1 averaged over entries rather than over links.

    Reported beside the micro average rather than instead of it, and it is the
    one that can see a rare entry. Micro weights every link equally, so the
    entries that fire often carry it -- and the entry that decides a case is
    frequently the one that fires on a handful of examples.
    """

    average: str = Param("macro")


@register_class(
    name="torchmetric",
    tags={"exact_set"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.ExactSetMatch",
)
class ExactSetMatchConfig(Configuration):
    """Share of examples whose named set is exactly the annotated one."""

    threshold: float = Param(0.5)
    ignore_index: int = Param(-1)


@register_class(
    name="torchmetric",
    tags={"empty_set"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.metrics.EmptySetAccuracy",
)
class EmptySetAccuracyConfig(ExactSetMatchConfig):
    """Share of examples annotated with nothing that were named nothing."""


class KnowledgeMetricConfig(MetricConfig):
    """A link metric reads the gate and the annotation behind it."""

    name: str = Param("link_f1")
    metric: RegistrationKey[Metric] = Param(LINK_F1)
    inputs: Sequence[str] = Param(["knowledge_mask", "knowledge_true"])


@register_class(
    name="metric",
    tags={"f1", "link"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class LinkF1MetricConfig(KnowledgeMetricConfig):
    pass


@register_class(
    name="metric",
    tags={"f1", "link", "macro"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class LinkMacroF1MetricConfig(KnowledgeMetricConfig):
    name: str = Param("link_macro_f1")
    metric: RegistrationKey[Metric] = Param(LINK_MACRO_F1)


@register_class(
    name="metric",
    tags={"exact_set"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class ExactSetMetricConfig(KnowledgeMetricConfig):
    name: str = Param("exact_set_match")
    metric: RegistrationKey[Metric] = Param(EXACT_SET_MATCH)


@register_class(
    name="metric",
    tags={"empty_set"},
    namespace=NAMESPACE,
    component=BOUND_METRIC_COMPONENT,
)
class EmptySetMetricConfig(KnowledgeMetricConfig):
    name: str = Param("empty_set_accuracy")
    metric: RegistrationKey[Metric] = Param(EMPTY_SET_ACCURACY)
