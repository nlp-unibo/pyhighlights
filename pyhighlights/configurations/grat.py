"""G-RAT: an attention guider regularizing the selector it is trained beside."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
    SPPPredictor,
)
from pyhighlights.components.models.spp.grat import GRATGuider
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    GUIDANCE_GUIDER,
    GUIDE_LOSS,
    JSD_LOSS,
    MLP_PREDICTOR,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_GUIDER,
)
from pyhighlights.utility.losses import Loss

ATTENTION_GUIDER_COMPONENT = "pyhighlights.components.models.spp.grat.AttentionGuider"
GRAT_COMPONENT = "pyhighlights.components.models.spp.grat.GRAT"
GUIDANCE_GUIDER_COMPONENT = "pyhighlights.components.models.spp.grat.GuidanceGuider"


@register_class(
    name="guider",
    tags={"attention", "gru"},
    namespace=NAMESPACE,
    component=ATTENTION_GUIDER_COMPONENT,
)
class AttentionGuiderConfig(Configuration):
    backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    noise_sigma: float = Param(1.0, ge=0.0)


@register_class(
    name="guider",
    tags={"attention", "transformer"},
    namespace=NAMESPACE,
    component=ATTENTION_GUIDER_COMPONENT,
)
class TransformerAttentionGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)


@register_class(
    name="guider",
    tags={"guidance", "gru"},
    namespace=NAMESPACE,
    component=GUIDANCE_GUIDER_COMPONENT,
)
class GuidanceGuiderConfig(Configuration):
    """The original G-RAT guider, sized as the library's GRU backbone."""

    vocab_size: int = Param(10_000, ge=3)
    embedding_dim: int = Param(128, ge=1)
    #: Per direction, as in ``GRUBackboneConfig``: the guider's states are
    #: twice as wide.
    hidden_size: int = Param(128, ge=1)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    noise_sigma: float = Param(1.0, ge=0.0)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)
    freeze_embeddings: bool | None = Param(None)


@register_class(
    name="model", tags={"grat", "gru"}, namespace=NAMESPACE, component=GRAT_COMPONENT
)
class GRUGRATConfig(SPPModelConfig):
    name: str = Param("grat")
    #: As the original implementation: no empty-selection repair. Its
    #: evaluation threshold of 0.5 is the argmax, so the selection is not
    #: sampled.
    repair_empty_selections: bool = Param(False)
    #: G-RAT encodes the highlight with its own backbone, so the predictor's
    #: is required rather than optional.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    guider: RegistrationKey[GRATGuider] = Param(GUIDANCE_GUIDER)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS, GUIDE_LOSS, JSD_LOSS]
    )
    guider_losses: List[RegistrationKey[Loss]] = Param([CLASSIFICATION_LOSS])
    pretrain_epochs: int = Param(10, ge=0)
    guide_decay: float = Param(1e-4, ge=0.0)
    #: One ``LayerNorm`` over the selector's and the predictor's states, as
    #: the original ``RNPSTModel`` has.
    share_layer_norm: bool = Param(True)
    guide_loss: str = Param("guide")
    jsd_loss: str = Param("jsd")


@register_class(
    name="model",
    tags={"grat", "transformer"},
    namespace=NAMESPACE,
    component=GRAT_COMPONENT,
)
class TransformerGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    guider: RegistrationKey[GRATGuider] = Param(TRANSFORMER_GUIDER)
    #: A transformer backbone has no normalisation of its states to share.
    share_layer_norm: bool = Param(False)
