"""FR: folded rationalization, one selector sharing the predictor backbone."""

from typing import TYPE_CHECKING

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    GRU_FR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_FR,
)

if TYPE_CHECKING:
    import pyhighlights.components.models.spp.base

FR_COMPONENT = "pyhighlights.components.models.spp.fr.FR"


@register_class(key=GRU_FR, component=FR_COMPONENT)
class GRUFRConfig(SPPModelConfig):
    name: str = Param("fr")
    sample_at_evaluation: bool = Param(
        True,
        description=(
            "Whether validation and test sample the hard Gumbel selection rather "
            "than take the argmax. FR samples, as its original implementation "
            "does."
        ),
    )
    repair_empty_selections: bool = Param(
        False,
        description=(
            "Whether a selection that kept no word keeps its highest-scoring "
            "word instead. FR does not repair, as its original implementation "
            "does not."
        ),
    )


@register_class(key=TRANSFORMER_FR, component=FR_COMPONENT)
class TransformerFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(TRANSFORMER_BACKBONE)
