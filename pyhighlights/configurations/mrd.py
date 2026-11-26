"""MRD: a generator trained to make the complement stop predicting the label."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
)
from pyhighlights.configurations.base import PhasedSPPModelConfig
from pyhighlights.configurations.keys import (
    COMPLEMENT_CLASSIFICATION_LOSS,
    FULL_CLASSIFICATION_LOSS,
    GRU_BACKBONE,
    GRU_MRD,
    REMAINING_DISCREPANCY_LOSS,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_MRD,
)
from pyhighlights.utility.losses import Loss

MRD_COMPONENT = "pyhighlights.components.models.spp.mrd.MRD"


@register_class(key=GRU_MRD, component=MRD_COMPONENT)
class GRUMRDConfig(PhasedSPPModelConfig):
    name: str = Param("mrd")
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes for the predictor, "
            "required because the predictor reads the complement and the full "
            "input with its own encoder."
        ),
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param(
        [COMPLEMENT_CLASSIFICATION_LOSS, FULL_CLASSIFICATION_LOSS],
        description=(
            "Registration keys of the losses scored in the predictor phase only. "
            "They read the complement and the full input and never the "
            "highlight, because the criterion of MRD is what the remainder can "
            "still say."
        ),
    )
    generator_losses: List[RegistrationKey[Loss]] = Param([REMAINING_DISCREPANCY_LOSS])


@register_class(key=TRANSFORMER_MRD, component=MRD_COMPONENT)
class TransformerMRDConfig(GRUMRDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
