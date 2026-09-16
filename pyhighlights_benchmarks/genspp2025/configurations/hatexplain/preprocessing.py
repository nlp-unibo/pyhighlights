"""What the corpus goes through before a model sees it.

A length filter, a folding of the labels, and a vote aggregation.
"""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.preprocessors import PipelineConfig
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_AGGREGATOR,
    HATEXPLAIN_LABEL_MAPPER,
    HATEXPLAIN_LENGTH_FILTER,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="preprocessor",
    tags={"hatexplain", "length"},
    namespace=NAMESPACE,
    component="pyhighlights.components.preprocessors.LengthFilter",
)
class LengthFilterConfig(Configuration):
    """Posts over thirty tokens are dropped, not truncated."""

    max_length: int = Param(30, ge=1)


@register_class(
    name="preprocessor",
    tags={"hatexplain", "labels"},
    namespace=NAMESPACE,
    component="pyhighlights.components.preprocessors.LabelMapper",
)
class LabelMapperConfig(Configuration):
    """``offensive`` becomes ``hatespeech``, before the votes are counted."""

    mapping: dict = Param({"offensive": "hatespeech"})
    column: str = Param("annotator_labels")


@register_class(
    name="preprocessor",
    tags={"aggregator", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.preprocessors.AnnotationAggregator",
)
class AggregatorConfig(Configuration):
    """Two classes left, so three annotators always have a majority."""

    labels: List[str] = Param(["hatespeech", "normal"])
    highlights: str = Param("majority")
    ties: str = Param("drop")


@register_class(
    name="preprocessor",
    tags={"hatexplain", "pipeline"},
    namespace=NAMESPACE,
    component="pyhighlights.components.preprocessors.Pipeline",
)
class HateXplainPipelineConfig(PipelineConfig):
    """Filter, fold, then reduce -- in that order."""

    steps: List[RegistrationKey[Preprocessor]] = Param(
        [HATEXPLAIN_LENGTH_FILTER, HATEXPLAIN_LABEL_MAPPER, HATEXPLAIN_AGGREGATOR]
    )
