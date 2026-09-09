"""FR: folded rationalization, one selector sharing the predictor backbone."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import NAMESPACE, TRANSFORMER_BACKBONE

FR_COMPONENT = "pyhighlights.components.models.spp.fr.FR"


@register_class(
    name="model", tags={"fr", "gru"}, namespace=NAMESPACE, component=FR_COMPONENT
)
class GRUFRConfig(SPPModelConfig):
    name: str = Param("fr")


@register_class(
    name="model",
    tags={"fr", "transformer"},
    namespace=NAMESPACE,
    component=FR_COMPONENT,
)
class TransformerFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
