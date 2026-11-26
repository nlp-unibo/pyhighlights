"""MCD: selected-input and full-input predictions trained in two phases."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
)
from pyhighlights.configurations.base import PhasedSPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    DISCREPANCY_LOSS,
    FULL_CLASSIFICATION_LOSS,
    GRU_MCD,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_MCD,
    UNNORMALISED_GRU_BACKBONE,
)
from pyhighlights.utility.losses import Loss

MCD_COMPONENT = "pyhighlights.components.models.spp.mcd.MCD"


@register_class(key=GRU_MCD, component=MCD_COMPONENT)
class GRUMCDConfig(PhasedSPPModelConfig):
    name: str = Param("mcd")
    sample_at_evaluation: bool = Param(
        True,
        description=(
            "Whether validation and test sample the hard Gumbel selection rather "
            "than take the argmax. MCD samples, as its original implementation "
            "does."
        ),
    )
    repair_empty_selections: bool = Param(
        False,
        description=(
            "Whether a selection that kept no word keeps its highest-scoring "
            "word instead. MCD does not repair, as its original implementation "
            "does not."
        ),
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        UNNORMALISED_GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes for the predictor, "
            "required because MCD reads the highlight and the full input with "
            "separate encoders. The GRU default is unnormalised, as the original "
            "implementation's predictor encoder is."
        ),
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, FULL_CLASSIFICATION_LOSS]
    )
    generator_losses: List[RegistrationKey[Loss]] = Param([DISCREPANCY_LOSS])


@register_class(key=TRANSFORMER_MCD, component=MCD_COMPONENT)
class TransformerMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
