"""The encoders: one GRU for the baselines, one for GenSPP."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class

from pyhighlights.configurations.backbones import GRUBackboneConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)

GRU_BACKBONE_COMPONENT = (
    "pyhighlights.components.models.spp.implementations.GRUBackbone"
)


@register_class(
    name="backbone",
    tags={"hatexplain"},
    namespace=NAMESPACE,
    component=GRU_BACKBONE_COMPONENT,
)
class HateXplainBackboneConfig(GRUBackboneConfig):
    """The baselines' encoder, embedding from the GloVe table the task reads."""

    #: A placeholder. The task reads GloVe and hands the matrix to the model,
    #: and :meth:`GRUBackbone.load_embeddings` **replaces** the table rather
    #: than copying into it, so the size declared here is never the size used.
    #: ``embedding_dim`` below is not a placeholder: the replacement is refused
    #: if its width disagrees.
    vocab_size: int = Param(2, ge=1)
    embedding_dim: int = Param(25, ge=1)
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)


@register_class(
    name="backbone",
    tags={"genspp", "hatexplain"},
    namespace=NAMESPACE,
    component=GRU_BACKBONE_COMPONENT,
)
class HateXplainGenSPPBackboneConfig(HateXplainBackboneConfig):
    bidirectional: bool = Param(False)
