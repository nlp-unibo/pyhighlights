"""The five architectures the paper puts on this corpus, and the parts they share."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone, SPPPredictor
from pyhighlights.configurations.backbones import MLPPredictorConfig
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig
from pyhighlights.configurations.grat import AttentionGuiderConfig, GRUGRATConfig
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.configurations.toy.keys import (
    TOY_BACKBONE,
    TOY_GENSPP_BACKBONE,
    TOY_GUIDER,
    TOY_PREDICTOR,
)


@register_class(
    name="predictor",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.MLPPredictor",
)
class ToyPredictorConfig(MLPPredictorConfig):
    num_classes: int = Param(3, ge=2)


@register_class(
    name="guider",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.AttentionGuider",
)
class ToyGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    noise_sigma: float = Param(1.0, ge=0.0)


@register_class(
    name="model",
    tags={"fr", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.fr.FR",
)
class ToyFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"mgr", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mgr.MGR",
)
class ToyMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TOY_BACKBONE, TOY_BACKBONE, TOY_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"mcd", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mcd.MCD",
)
class ToyMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"grat", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.GRAT",
)
class ToyGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    guider: RegistrationKey = Param(TOY_GUIDER)
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)


@register_class(
    name="model",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPP",
)
class ToyGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
