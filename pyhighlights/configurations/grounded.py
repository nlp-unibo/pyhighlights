"""Grounded SPP: the comparer, and a model that reads a knowledge base."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.components.models.spp.grounded import SPPComparer
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    ENTAILMENT_COMPARER,
    KNOWLEDGE_LOSS,
    KNOWLEDGE_SPARSITY_LOSS,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss

GROUNDED_COMPONENT = "pyhighlights.components.models.spp.grounded.GroundedSPP"


@register_class(
    name="comparer",
    tags={"entailment"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grounded.EntailmentComparer",
)
class EntailmentComparerConfig(Configuration):
    """A directed judgement over ``[u; v; u - v; u * v]``.

    ``input_size`` is not a parameter: the model passes its backbone's width,
    the same way a selector and a predictor are sized.
    """

    hidden_sizes: List[int] = Param([128])


@register_class(
    name="model",
    tags={"grounded", "gru"},
    namespace=NAMESPACE,
    component=GROUNDED_COMPONENT,
)
class GRUGroundedConfig(SPPModelConfig):
    """Select-then-predict grounded in a corpus's knowledge base.

    Four terms rather than three. The classification and the two readability
    penalties are the ones every SPP model carries, over the union of the
    per-entry highlights. The fourth scores which entries were named, against
    the gold links, and it is what separates this from an unsupervised run:
    without it the comparer learns how often each entry fires and stops
    reading its inputs.

    The knowledge sparsity term is registered and left out of the default
    list: its target is the mean share of a base that one example
    instantiates, which differs per corpus by a factor of four, so a study
    that wants it states its own threshold rather than inheriting a number
    that fits nothing.
    """

    name: str = Param("grounded")
    comparer: RegistrationKey[SPPComparer] = Param(ENTAILMENT_COMPARER)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS, KNOWLEDGE_LOSS]
    )


@register_class(
    name="model",
    tags={"grounded", "transformer"},
    namespace=NAMESPACE,
    component=GROUNDED_COMPONENT,
)
class TransformerGroundedConfig(GRUGroundedConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)


__all__ = [
    "EntailmentComparerConfig",
    "GRUGroundedConfig",
    "KNOWLEDGE_SPARSITY_LOSS",
    "TransformerGroundedConfig",
]
