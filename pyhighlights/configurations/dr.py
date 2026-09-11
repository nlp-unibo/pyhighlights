"""DR: a predictor trained at the rate of the selection it is given."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    GRU_BACKBONE,
    NAMESPACE,
    TRANSFORMER_BACKBONE,
)

DR_COMPONENT = "pyhighlights.components.models.spp.dr.DR"


@register_class(
    name="model", tags={"dr", "gru"}, namespace=NAMESPACE, component=DR_COMPONENT
)
class GRUDRConfig(SPPModelConfig):
    name: str = Param("dr")
    #: Separate encoders: DR decouples the two rates, and a shared encoder
    #: would take both of them.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    #: The floor under the predictor's rate, as a fraction of the selector's.
    #: The paper's value. Below it a sparse selection stops the predictor
    #: outright, and a predictor that never moves teaches the selector nothing.
    scale_floor: float = Param(0.05, gt=0.0, le=1.0)


@register_class(
    name="model",
    tags={"dr", "transformer"},
    namespace=NAMESPACE,
    component=DR_COMPONENT,
)
class TransformerDRConfig(GRUDRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
