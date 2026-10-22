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
    #: As the original implementation: a sampled selection at evaluation,
    #: pooling over every valid position, and no empty-selection repair.
    sample_at_evaluation: bool = Param(True)
    pool_dropped: bool = Param(True)
    repair_empty_selections: bool = Param(False)


@register_class(
    name="model",
    tags={"fr", "transformer"},
    namespace=NAMESPACE,
    component=FR_COMPONENT,
)
class TransformerFRConfig(GRUFRConfig):
    #: A transformer's dropped positions are masked out of its attention, so
    #: their states carry nothing to pool. No original pools them.
    pool_dropped: bool = Param(False)
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
