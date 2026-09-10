"""G-RAT: an attention guider regularizing the selector it is trained beside."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPAggregator,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.components.models.spp.grat import GRATGuider
from pyhighlights.configurations.keys import (
    ADAM,
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    GRU_GUIDER,
    GUIDE_LOSS,
    JSD_LOSS,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_GUIDER,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

ATTENTION_GUIDER_COMPONENT = "pyhighlights.components.models.spp.grat.AttentionGuider"
GRAT_COMPONENT = "pyhighlights.components.models.spp.grat.GRAT"


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
    name="model", tags={"grat", "gru"}, namespace=NAMESPACE, component=GRAT_COMPONENT
)
class GRUGRATConfig(Configuration):
    name: str = Param("grat")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    guider: RegistrationKey[GRATGuider] = Param(GRU_GUIDER)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    select_over: str = Param("word", variants=["subtoken"])
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS, GUIDE_LOSS, JSD_LOSS]
    )
    guider_losses: List[RegistrationKey[Loss]] = Param([CLASSIFICATION_LOSS])
    pretrain_epochs: int = Param(10, ge=0)
    guide_decay: float = Param(1e-4, ge=0.0)
    guide_loss: str = Param("guide")
    jsd_loss: str = Param("jsd")
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    test_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)


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
