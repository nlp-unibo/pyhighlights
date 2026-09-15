"""The encoders: one GRU for the baselines, one for GenSPP."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class

from pyhighlights.configurations.backbones import GRUBackboneConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.configurations.toy import (
    EMBEDDING_DIM,
    VOCABULARY_SIZE,
)

GRU_BACKBONE_COMPONENT = (
    "pyhighlights.components.models.spp.implementations.GRUBackbone"
)


@register_class(
    name="backbone", tags={"toy"}, namespace=NAMESPACE, component=GRU_BACKBONE_COMPONENT
)
class ToyBackboneConfig(GRUBackboneConfig):
    """The baselines' encoder, reading a one-hot table of the alphabet."""

    vocab_size: int = Param(VOCABULARY_SIZE, ge=1)
    embedding_dim: int = Param(EMBEDDING_DIM, ge=1)
    hidden_size: int = Param(8, ge=1)
    freeze_embeddings: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)


@register_class(
    name="backbone",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component=GRU_BACKBONE_COMPONENT,
)
class ToyGenSPPBackboneConfig(ToyBackboneConfig):
    """The genetic half's encoder: one direction, the same table.

    The release declares 26 columns here against the baselines' 25 for the
    same twenty-four characters. Both are the alphabet's width plus dead
    columns, so both halves read :data:`EMBEDDING_DIM`.
    """

    bidirectional: bool = Param(False)
