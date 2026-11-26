"""G-RAT: an attention guider regularizing the selector it is trained beside."""

from typing import TYPE_CHECKING, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    GRU_GRAT,
    GRU_GUIDER,
    GUIDANCE_GUIDER,
    GUIDE_LOSS,
    JSD_LOSS,
    MLP_PREDICTOR,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_GRAT,
    TRANSFORMER_GUIDER,
)

if TYPE_CHECKING:
    import pyhighlights.components.models.spp.base
    import pyhighlights.components.models.spp.grat
    import pyhighlights.utility.losses

ATTENTION_GUIDER_COMPONENT = "pyhighlights.components.models.spp.grat.AttentionGuider"
GRAT_COMPONENT = "pyhighlights.components.models.spp.grat.GRAT"
GUIDANCE_GUIDER_COMPONENT = "pyhighlights.components.models.spp.grat.GuidanceGuider"


@register_class(key=GRU_GUIDER, component=ATTENTION_GUIDER_COMPONENT)
class AttentionGuiderConfig(Configuration):
    backbone: RegistrationKey["pyhighlights.components.models.spp.base.SPPBackbone"] = (
        Param(
            GRU_BACKBONE,
            description=(
                "Registration key of the backbone the guider encodes the input with."
            ),
        )
    )
    predictor: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPPredictor"
    ] = Param(
        MLP_PREDICTOR,
        description=(
            "Registration key of the predictor that maps the guider's "
            "attention-pooled state to class logits."
        ),
    )
    noise_sigma: float = Param(
        1.0,
        ge=0.0,
        description=(
            "Scale of the noise ``|N(0, noise_sigma)|`` added to the guider's "
            "attention scores in training. Zero adds no noise."
        ),
    )


@register_class(key=TRANSFORMER_GUIDER, component=ATTENTION_GUIDER_COMPONENT)
class TransformerAttentionGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey["pyhighlights.components.models.spp.base.SPPBackbone"] = (
        Param(TRANSFORMER_BACKBONE)
    )


@register_class(key=GUIDANCE_GUIDER, component=GUIDANCE_GUIDER_COMPONENT)
class GuidanceGuiderConfig(Configuration):
    """The original G-RAT guider, sized as the library's GRU backbone."""

    vocabulary_size: int = Param(
        10_000,
        ge=3,
        description=(
            "Rows the guider's embedding table has, which is the task's "
            "``vocabulary_size``."
        ),
    )
    embedding_dim: int = Param(128, ge=1, description="Width of each token embedding.")
    hidden_size: int = Param(
        128,
        ge=1,
        description=(
            "Width of the guider's GRU states per direction, as in "
            "``GRUBackboneConfig``, so the guider's states are twice as wide."
        ),
    )
    predictor: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPPredictor"
    ] = Param(
        MLP_PREDICTOR,
        description=(
            "Registration key of the predictor that maps the guider's pooled "
            "state to class logits."
        ),
    )
    noise_sigma: float = Param(
        1.0,
        ge=0.0,
        description=(
            "Scale of the noise ``|N(0, noise_sigma)|`` added to the guider's "
            "attention scores in training. Zero adds no noise."
        ),
    )
    dropout_rate: float = Param(
        0.0,
        ge=0.0,
        lt=1.0,
        description=(
            "Dropout probability applied to the states before they are scored "
            "for attention and to the pooled state before the predictor."
        ),
    )
    freeze_embeddings: bool | None = Param(
        None,
        description=(
            "``None`` trains a randomly initialised table and freezes loaded "
            "vectors. ``True`` freezes either table, and ``False`` trains either."
        ),
    )


@register_class(key=GRU_GRAT, component=GRAT_COMPONENT)
class GRUGRATConfig(SPPModelConfig):
    name: str = Param("grat")
    repair_empty_selections: bool = Param(
        False,
        description=(
            "Whether a selection that kept no word keeps its highest-scoring "
            "one. Off as in the original implementation, whose evaluation "
            "threshold of 0.5 is the argmax, so the selection is not sampled."
        ),
    )
    predictor_backbone: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes the highlight for "
            "the predictor. Required rather than optional, because G-RAT "
            "encodes the highlight with its own backbone."
        ),
    )
    guider: RegistrationKey["pyhighlights.components.models.spp.grat.GRATGuider"] = (
        Param(
            GUIDANCE_GUIDER,
            description=(
                "Registration key of the guider, the attention classifier over the "
                "full input whose attention guides the selector."
            ),
        )
    )
    losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS, GUIDE_LOSS, JSD_LOSS]
    )
    guider_losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [CLASSIFICATION_LOSS],
        description="Registration keys of the losses that train the guider alone.",
    )
    pretrain_epochs: int = Param(
        10,
        ge=0,
        description=(
            "Number of epochs the guider trains alone before the selector and "
            "the predictor take their first step."
        ),
    )
    guide_decay: float = Param(
        1e-4,
        ge=0.0,
        description=(
            "Amount the guide term's weight falls per training step of the "
            "selector and the predictor, from one down to zero. The JSD term "
            "takes the rest of the weight."
        ),
    )
    share_layer_norm: bool = Param(
        True,
        description=(
            "Whether one ``LayerNorm`` normalises both the selector's and the "
            "predictor's states, as the original ``RNPSTModel`` has."
        ),
    )
    guide_loss: str = Param(
        "guide",
        description=(
            "Name of the loss that the guide term's weight scales. It has to "
            "name one of the model's losses."
        ),
    )
    jsd_loss: str = Param(
        "jsd",
        description=(
            "Name of the loss scaled by one minus the guide term's weight. It "
            "has to name one of the model's losses."
        ),
    )


@register_class(key=TRANSFORMER_GRAT, component=GRAT_COMPONENT)
class TransformerGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(TRANSFORMER_BACKBONE)
    guider: RegistrationKey["pyhighlights.components.models.spp.grat.GRATGuider"] = (
        Param(TRANSFORMER_GUIDER)
    )
    share_layer_norm: bool = Param(
        False,
        description=(
            "Whether one ``LayerNorm`` normalises both the selector's and the "
            "predictor's states. Off, because a transformer backbone has no "
            "normalisation of its states to share."
        ),
    )
