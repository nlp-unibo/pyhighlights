"""Backbone, selector and predictor registrations shared by the SPP models."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import (
    FROZEN_TRANSFORMER_BACKBONE,
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    SIGMOID_SELECTOR,
    STACKED_BACKBONE,
    TRANSFORMER_BACKBONE,
    UNNORMALISED_GRU_BACKBONE,
)


@register_class(
    key=GRU_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.GRUBackbone",
)
class GRUBackboneConfig(Configuration):
    vocabulary_size: int = Param(
        10_000,
        ge=3,
        description=(
            "Rows the embedding table has, which is the task's ``vocabulary_size``: "
            "id 0 is padding, id 1 is the unknown token, and words start at 2."
        ),
    )
    embedding_dim: int = Param(128, ge=1, description="Width of each token embedding.")
    hidden_size: int = Param(
        128,
        ge=1,
        description=(
            "Width of the GRU states per direction, so a bidirectional "
            "backbone outputs twice this width."
        ),
    )
    freeze_embeddings: bool | None = Param(
        None,
        description=(
            "``None`` trains a randomly initialised table and freezes loaded "
            "vectors. ``True`` freezes either table, and ``False`` trains either."
        ),
    )
    num_layers: int = Param(1, ge=1, description="Number of stacked GRU layers.")
    bidirectional: bool = Param(
        True, description="Whether the GRU reads the sequence in both directions."
    )
    dropout_rate: float = Param(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied to the recurrent states.",
    )
    layer_norm: bool = Param(
        True,
        description=(
            "Whether layer normalisation is applied to the recurrent states "
            "before dropout."
        ),
    )


@register_class(
    key=UNNORMALISED_GRU_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.GRUBackbone",
)
class UnnormalisedGRUBackboneConfig(GRUBackboneConfig):
    """The GRU backbone without layer normalisation.

    The predictors of the original MCD and MGR encode this way, while their
    generators normalise.
    """

    layer_norm: bool = Param(False)


@register_class(
    key=TRANSFORMER_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
)
class TransformerBackboneConfig(Configuration):
    pretrained_model_card: str = Param(
        "distilbert-base-uncased",
        description="Hugging Face model card of the pretrained encoder.",
    )
    num_features: int | None = Param(
        None,
        ge=1,
        description=(
            "Number of token embeddings the encoder is resized to, or ``None`` "
            "to keep the pretrained vocabulary."
        ),
    )
    freeze_transformer: bool = Param(
        False,
        description=(
            "Whether the encoder's weights are held fixed. A frozen encoder "
            "stays in evaluation mode, so its dropout never runs."
        ),
    )


@register_class(
    key=FROZEN_TRANSFORMER_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
)
class FrozenTransformerBackboneConfig(TransformerBackboneConfig):
    """The same encoder with its weights held, as a key rather than an argument.

    A model names a backbone key, and build arguments reach the component a
    caller builds, not the ones built underneath it. ``freeze_transformer``
    is therefore set by this key rather than passed from the outside.

    A frozen encoder is a different experiment, not a cheaper approximation of
    the same one: the selector reads representations nothing adapted to its
    task. Two places it is the right one: an ablation, where a fine-tuned
    encoder would absorb the difference being measured, and a run whose size
    is bounded by memory rather than by patience. A frozen encoder holds no
    gradient, no gradient buffer and no optimizer state for its parameters.
    """

    freeze_transformer: bool = Param(True)


@register_class(
    key=STACKED_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.StackedBackbone",
)
class StackedBackboneConfig(Configuration):
    """A frozen transformer read by a GRU trained from scratch.

    The reference implementations of FR, MCD, MRD, DR, MGR and DAR use a
    bidirectional GRU over a frozen pretrained table. This backbone replaces
    the table with a transformer. While ``freeze_transformer`` holds, one
    learning rate serves it, because everything trainable starts from scratch.
    Fine-tuning the transformer needs ``encoder_lr``.

    ``hidden_size`` is the library's GRU default rather than the transformer's
    width: the GRU is the encoder here, and 768 units of it is a large layer to
    train on a corpus of a few thousand documents.
    """

    pretrained_model_card: str = Param(
        "distilbert-base-uncased",
        description="Hugging Face model card of the pretrained transformer.",
    )
    hidden_size: int = Param(
        128,
        ge=1,
        description=(
            "Width of the GRU states per direction, so a bidirectional "
            "backbone outputs twice this width."
        ),
    )
    num_features: int | None = Param(
        None,
        ge=1,
        description=(
            "Number of token embeddings the transformer is resized to, or "
            "``None`` to keep the pretrained vocabulary."
        ),
    )
    freeze_transformer: bool = Param(
        True,
        description=(
            "Whether the transformer's weights are held fixed. Fine-tuning it "
            "needs ``encoder_lr``, because the GRU above it trains from scratch."
        ),
    )
    num_layers: int = Param(1, ge=1, description="Number of stacked GRU layers.")
    bidirectional: bool = Param(
        True, description="Whether the GRU reads the sequence in both directions."
    )
    dropout_rate: float = Param(
        0.0,
        ge=0.0,
        lt=1.0,
        description="Dropout probability applied to the recurrent states.",
    )


@register_class(
    key=MLP_SELECTOR,
    component="pyhighlights.components.models.spp.implementations.MLPSelector",
)
class MLPSelectorConfig(Configuration):
    hidden_sizes: List[int] = Param(
        [],
        description=(
            "Widths of the hidden layers between the states and the two "
            "selector outputs, with a GELU between layers. An empty list is one "
            "linear layer."
        ),
    )


@register_class(
    key=SIGMOID_SELECTOR,
    component="pyhighlights.components.models.spp.implementations.SigmoidSelector",
)
class SigmoidSelectorConfig(Configuration):
    hidden_sizes: List[int] = Param(
        [],
        description=(
            "Widths of the hidden layers between the states and the single "
            "selector output, with a GELU between layers. An empty list is one "
            "linear layer."
        ),
    )


@register_class(
    key=MLP_PREDICTOR,
    component="pyhighlights.components.models.spp.implementations.MLPPredictor",
)
class MLPPredictorConfig(Configuration):
    hidden_sizes: List[int] = Param(
        [],
        description=(
            "Widths of the hidden layers between the pooled state and the class "
            "logits, with a GELU between layers. An empty list is one linear "
            "layer."
        ),
    )
    num_classes: int = Param(
        2, ge=2, description="Number of classes the predictor outputs logits for."
    )
