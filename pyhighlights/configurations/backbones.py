"""Backbone, selector and predictor registrations shared by the SPP models."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_method

#: Namespace cinnamon resolves this module's registrations under. Kept a literal
#: in every registering module: ``NamespaceExtractor`` reads it statically and only
#: sees bindings made in the same file.
NAMESPACE = "pyhighlights"


class GRUBackboneConfig(Configuration):
    vocab_size: int = Param(10_000, ge=1)
    embedding_dim: int = Param(128, ge=1)
    hidden_size: int = Param(128, ge=1)
    freeze_embeddings: bool = Param(False)
    num_layers: int = Param(1, ge=1)
    bidirectional: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)

    @classmethod
    @register_method(
        name="backbone",
        tags={"gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class TransformerBackboneConfig(Configuration):
    pretrained_model_card: str = Param("distilbert-base-uncased")
    num_features: int | None = Param(None, ge=1)
    freeze_transformer: bool = Param(False)

    @classmethod
    @register_method(
        name="backbone",
        tags={"transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
    )
    def default(cls):
        return super().default()


class MLPSelectorConfig(Configuration):
    hidden_sizes: List[int] = Param([])

    @classmethod
    @register_method(
        name="selector",
        tags={"mlp"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.MLPSelector",
    )
    def default(cls):
        return super().default()


class MLPPredictorConfig(Configuration):
    hidden_sizes: List[int] = Param([])
    num_classes: int = Param(2, ge=2)

    @classmethod
    @register_method(
        name="predictor",
        tags={"mlp"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.MLPPredictor",
    )
    def default(cls):
        return super().default()


class GenSPPGRUBackboneConfig(GRUBackboneConfig):
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    bidirectional: bool = Param(False)

    @classmethod
    @register_method(
        name="backbone",
        tags={"genspp", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class GenSPPTransformerBackboneConfig(TransformerBackboneConfig):
    freeze_transformer: bool = Param(True)

    @classmethod
    @register_method(
        name="backbone",
        tags={"genspp", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
    )
    def default(cls):
        return super().default()
