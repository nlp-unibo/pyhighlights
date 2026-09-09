"""Leakage detection and preprocessing registrations."""

from typing import List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.loaders import HateXplainLoader
from pyhighlights.components.preprocessors import PRIORITY, Preprocessor
from pyhighlights.configurations.keys import (
    HATEXPLAIN_AGGREGATOR,
    LEAKAGE_REMOVER,
    NAMESPACE,
)


class LeakageDetectorConfig(Configuration):
    """Reports what splits share; raises above ``tolerance``."""

    key: str = Param("text")
    tolerance: float = Param(0.0, ge=0.0, le=1.0)
    normalize_keys: bool = Param(True)

    @classmethod
    @register_method(
        name="detector",
        tags={"leakage"},
        namespace=NAMESPACE,
        component="pyhighlights.components.leakage.LeakageDetector",
    )
    def default(cls):
        return super().default()


class LeakageRemoverConfig(Configuration):
    """Drops shared and repeated rows, protecting the annotated split first."""

    priority: Sequence[str] = Param(list(PRIORITY))
    key: str = Param("text")
    normalize_keys: bool = Param(True)

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"leakage"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.LeakageRemover",
    )
    def default(cls):
        return super().default()


class HateXplainAggregatorConfig(Configuration):
    """Three annotators to one label and one highlight vector."""

    labels: Sequence[str] = Param(list(HateXplainLoader.LABELS))
    rationale: str = Param("majority", variants=["union", "intersection"])
    ties: str = Param("drop", variants=["keep"])

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"aggregator", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.AnnotationAggregator",
    )
    def default(cls):
        return super().default()


class ClassWeightsConfig(Configuration):
    """Reads one split's class frequencies; changes no row."""

    split: str = Param("train")
    #: Left unset, the number of classes is the largest label seen plus one.
    #: Set it wherever a split might not hold every class.
    classes: int | None = Param(None)

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"class_weights"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.ClassWeights",
    )
    def default(cls):
        return super().default()


class PipelineConfig(Configuration):
    """Preprocessors run in order, each over what the last returned."""

    steps: List[RegistrationKey[Preprocessor]] = Param([LEAKAGE_REMOVER])

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"pipeline"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.Pipeline",
    )
    def default(cls):
        return super().default()


class HateXplainPipelineConfig(PipelineConfig):
    """Aggregate the annotators first: repairing leakage before the tie-drop
    would measure overlap over rows the aggregation then removes."""

    steps: List[RegistrationKey[Preprocessor]] = Param(
        [HATEXPLAIN_AGGREGATOR, LEAKAGE_REMOVER]
    )

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"pipeline", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.Pipeline",
    )
    def default(cls):
        return super().default()


__all__: List[str] = [
    "ClassWeightsConfig",
    "HateXplainAggregatorConfig",
    "HateXplainPipelineConfig",
    "LeakageDetectorConfig",
    "LeakageRemoverConfig",
    "PipelineConfig",
]
