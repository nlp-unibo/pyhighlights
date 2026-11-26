"""DR: a predictor trained at the rate of the selection it is given."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    GRU_BACKBONE,
    GRU_DR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_DR,
)

DR_COMPONENT = "pyhighlights.components.models.spp.dr.DR"


@register_class(key=GRU_DR, component=DR_COMPONENT)
class GRUDRConfig(SPPModelConfig):
    name: str = Param("dr")
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes for the predictor, "
            "separate from the selector's because DR decouples the two learning "
            "rates and a shared encoder would take both of them."
        ),
    )
    scale_floor: float = Param(
        0.05,
        gt=0.0,
        le=1.0,
        description=(
            "Lower bound, in (0, 1], on the predictor's learning rate as a "
            "fraction of the selector's, set to the reference implementation's "
            "value. Below it a sparse selection stops the predictor outright, "
            "and a predictor that never moves teaches the selector nothing."
        ),
    )


@register_class(key=TRANSFORMER_DR, component=DR_COMPONENT)
class TransformerDRConfig(GRUDRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
