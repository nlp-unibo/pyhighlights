"""Backbone, selector and predictor registrations shared by the SPP models."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import NAMESPACE


@register_class(
    name="backbone",
    tags={"gru"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.GRUBackbone",
)
class GRUBackboneConfig(Configuration):
    vocab_size: int = Param(10_000, ge=1)
    embedding_dim: int = Param(128, ge=1)
    hidden_size: int = Param(128, ge=1)
    freeze_embeddings: bool = Param(False)
    num_layers: int = Param(1, ge=1)
    bidirectional: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)


@register_class(
    name="backbone",
    tags={"transformer"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
)
class TransformerBackboneConfig(Configuration):
    pretrained_model_card: str = Param("distilbert-base-uncased")
    num_features: int | None = Param(None, ge=1)
    freeze_transformer: bool = Param(False)


@register_class(
    name="selector",
    tags={"mlp"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.MLPSelector",
)
class MLPSelectorConfig(Configuration):
    hidden_sizes: List[int] = Param([])


@register_class(
    name="predictor",
    tags={"mlp"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.MLPPredictor",
)
class MLPPredictorConfig(Configuration):
    hidden_sizes: List[int] = Param([])
    num_classes: int = Param(2, ge=2)
