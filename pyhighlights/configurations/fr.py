"""FR: folded rationalization, one selector sharing the predictor backbone."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    GRU_FR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_FR,
)

FR_COMPONENT = "pyhighlights.components.models.spp.fr.FR"


@register_class(key=GRU_FR, component=FR_COMPONENT)
class GRUFRConfig(SPPModelConfig):
    name: str = Param("fr")
    #: As the original implementation: a sampled selection at evaluation and
    #: no empty-selection repair.
    sample_at_evaluation: bool = Param(True)
    repair_empty_selections: bool = Param(False)


@register_class(key=TRANSFORMER_FR, component=FR_COMPONENT)
class TransformerFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
