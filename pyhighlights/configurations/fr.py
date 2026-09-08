"""FR: folded rationalization, one selector sharing the predictor backbone."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import NAMESPACE, TRANSFORMER_BACKBONE


class GRUFRConfig(SPPModelConfig):
    name: str = Param("fr")

    @classmethod
    @register_method(
        name="model",
        tags={"fr", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.fr.FR",
    )
    def default(cls):
        return super().default()


class TransformerFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)

    @classmethod
    @register_method(
        name="model",
        tags={"fr", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.fr.FR",
    )
    def default(cls):
        return super().default()
