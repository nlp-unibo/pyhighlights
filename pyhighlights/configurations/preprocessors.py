"""Corpus audits and preprocessing registrations."""

from typing import List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.keys import (
    CLASS_WEIGHTS,
    HATEXPLAIN_AGGREGATOR,
    HATEXPLAIN_PIPELINE,
    KNOWLEDGE_WEIGHTS,
    LEAKAGE_DETECTOR,
    LEAKAGE_REMOVER,
    PIPELINE,
    SHORTCUT_DETECTOR,
)
from pyhighlights.constants import HATEXPLAIN_LABELS, PRIORITY

PIPELINE_COMPONENT = "pyhighlights.components.preprocessors.Pipeline"


@register_class(
    key=LEAKAGE_DETECTOR, component="pyhighlights.components.leakage.LeakageDetector"
)
class LeakageDetectorConfig(Configuration):
    """Reports what splits share, and refuses a corpus that shares anything."""

    key: str = Param("text", description="Column whose value identifies a row.")
    normalize_keys: bool = Param(
        True,
        description=(
            "Whether keys are compared with whitespace collapsed, stripped and "
            "lower-cased, rather than exactly as the corpus spells them."
        ),
    )


@register_class(
    key=SHORTCUT_DETECTOR,
    component="pyhighlights.components.shortcuts.ShortcutDetector",
)
class ShortcutDetectorConfig(Configuration):
    """Reports what predicts the label, and refuses a corpus solvable without
    its evidence."""

    max_length: int = Param(
        4,
        ge=1,
        description=(
            "Longest n-gram scanned, in tokens, at least 1. Every shorter one is "
            "scanned too."
        ),
    )
    separator: str = Param(
        "",
        description=(
            "String that joins an n-gram's tokens for display: empty for a corpus "
            "of characters, a space for a corpus of words."
        ),
    )
    tokens: str = Param(
        "tokens", description="Column holding the token sequence of each example."
    )
    label: str = Param("label", description="Column holding the class label.")
    seed: int = Param(0, description="Random seed of the label shuffles.")
    permutations: int = Param(
        30,
        ge=1,
        description=(
            "Number of label shuffles the refusal threshold is the best of, which "
            "makes the check a permutation test at a level of about "
            "``1 / permutations``."
        ),
    )


@register_class(
    key=LEAKAGE_REMOVER,
    component="pyhighlights.components.preprocessors.LeakageRemover",
)
class LeakageRemoverConfig(Configuration):
    """Drops shared and repeated rows, protecting the annotated split first."""

    priority: Sequence[str] = Param(
        list(PRIORITY),
        description=(
            "Split names from the one that must stay intact to the one that can "
            "afford to lose rows; splits not listed come last. The annotated split "
            "comes first by default, since it is the only one carrying highlights."
        ),
    )
    key: str = Param("text", description="Column whose value identifies a row.")
    normalize_keys: bool = Param(
        True,
        description=(
            "Whether keys are compared with whitespace collapsed, stripped and "
            "lower-cased, rather than exactly as the corpus spells them."
        ),
    )


@register_class(
    key=HATEXPLAIN_AGGREGATOR,
    component="pyhighlights.components.preprocessors.AnnotationAggregator",
)
class HateXplainAggregatorConfig(Configuration):
    """Three annotators to one label and one highlight vector."""

    labels: Sequence[str] = Param(
        list(HATEXPLAIN_LABELS),
        description=(
            "Label names in class index order, which map each annotator vote to "
            "its class index."
        ),
    )
    highlights: str = Param(
        "majority",
        variants=["union", "intersection"],
        description=(
            "How annotator highlights combine: ``majority`` marks a token more "
            "than half the annotators marked, ``union`` any, ``intersection`` all."
        ),
    )
    ties: str = Param(
        "drop",
        variants=["keep"],
        description=(
            "What happens to a post whose top label is shared with another: "
            "``drop`` removes it, as the HateXplain paper does, and ``keep`` "
            "resolves it by annotator order."
        ),
    )


@register_class(
    key=CLASS_WEIGHTS, component="pyhighlights.components.preprocessors.ClassWeights"
)
class ClassWeightsConfig(Configuration):
    """Reads one split's class frequencies; changes no row."""

    split: str = Param("train", description="Split whose class frequencies are read.")
    classes: int | None = Param(
        None,
        ge=1,
        description=(
            "Number of classes, at least 1; unset means the largest label seen "
            "plus one. Set it wherever a split might not hold every class."
        ),
    )


@register_class(
    key=KNOWLEDGE_WEIGHTS,
    component="pyhighlights.components.preprocessors.KnowledgeWeights",
)
class KnowledgeWeightsConfig(Configuration):
    """Reads one split's example-to-entry links; changes no row.

    ``ClassWeightsTask`` runs it unchanged, so the weights land in a
    ``results.json`` a configuration can be copied from rather than being
    computed inside training and left nowhere.
    """

    split: str = Param(
        "train", description="Split whose example-to-entry links are read."
    )
    entries: int = Param(
        2,
        ge=1,
        description=(
            "Number of entries in the knowledge base, at least 1, stated rather "
            "than inferred: the largest index a split happens to use is not how "
            "many entries there are."
        ),
    )


@register_class(key=PIPELINE, component=PIPELINE_COMPONENT)
class PipelineConfig(Configuration):
    """Preprocessors run in order, each over what the last returned."""

    steps: List[RegistrationKey[Preprocessor]] = Param(
        [LEAKAGE_REMOVER],
        description=(
            "Registration keys of the preprocessors, run in order, each over what "
            "the last returned."
        ),
    )


@register_class(key=HATEXPLAIN_PIPELINE, component=PIPELINE_COMPONENT)
class HateXplainPipelineConfig(PipelineConfig):
    """Aggregate the annotators first: repairing leakage before the tie-drop
    would measure overlap over rows the aggregation then removes."""

    steps: List[RegistrationKey[Preprocessor]] = Param(
        [HATEXPLAIN_AGGREGATOR, LEAKAGE_REMOVER]
    )


__all__: List[str] = [
    "ClassWeightsConfig",
    "HateXplainAggregatorConfig",
    "HateXplainPipelineConfig",
    "KnowledgeWeightsConfig",
    "LeakageDetectorConfig",
    "LeakageRemoverConfig",
    "PipelineConfig",
    "ShortcutDetectorConfig",
]
