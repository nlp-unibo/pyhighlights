"""The five architectures the paper puts on this corpus, and the parts they share."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig
from pyhighlights.configurations.grat import AttentionGuiderConfig, GRUGRATConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    DISCREPANCY_LOSS,
    FULL_CLASSIFICATION_LOSS,
)
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights.utility.losses import Loss
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_BACKBONE,
    HATEXPLAIN_GENSPP_BACKBONE,
    HATEXPLAIN_GUIDE_LOSS,
    HATEXPLAIN_GUIDER,
    HATEXPLAIN_JSD_LOSS,
    HATEXPLAIN_SPARSITY_LOSS,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="guider",
    tags={"hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.AttentionGuider",
)
class HateXplainGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    noise_sigma: float = Param(1.0, ge=0.0)


@register_class(
    name="model",
    tags={"fr", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.fr.FR",
)
class HateXplainFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )


@register_class(
    name="model",
    tags={"hatexplain", "mgr"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mgr.MGR",
)
class HateXplainMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )


@register_class(
    name="model",
    tags={"hatexplain", "mcd"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mcd.MCD",
)
class HateXplainMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    shared_losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param([FULL_CLASSIFICATION_LOSS])
    generator_losses: List[RegistrationKey[Loss]] = Param([DISCREPANCY_LOSS])


@register_class(
    name="model",
    tags={"grat", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.GRAT",
)
class HateXplainGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    guider: RegistrationKey = Param(HATEXPLAIN_GUIDER)
    losses: List[RegistrationKey[Loss]] = Param(
        [
            CLASSIFICATION_LOSS,
            HATEXPLAIN_SPARSITY_LOSS,
            HATEXPLAIN_GUIDE_LOSS,
            HATEXPLAIN_JSD_LOSS,
        ]
    )
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)


@register_class(
    name="model",
    tags={"genspp", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPP",
)
class HateXplainGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)
