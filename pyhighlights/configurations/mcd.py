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
    NAMESPACE,
    TRANSFORMER_BACKBONE,
    UNNORMALISED_GRU_BACKBONE,
)
from pyhighlights.utility.losses import Loss

MCD_COMPONENT = "pyhighlights.components.models.spp.mcd.MCD"


@register_class(
    name="model", tags={"gru", "mcd"}, namespace=NAMESPACE, component=MCD_COMPONENT
)
class GRUMCDConfig(PhasedSPPModelConfig):
    name: str = Param("mcd")
    #: As the original implementation: a sampled selection at evaluation and
    #: no empty-selection repair.
    sample_at_evaluation: bool = Param(True)
    repair_empty_selections: bool = Param(False)
    #: MCD reads the highlight and the full input with separate encoders, so
    #: the predictor's backbone is required rather than optional.
    #: Unnormalised, as the original implementation's predictor encoder.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(UNNORMALISED_GRU_BACKBONE)
    predictor_losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, FULL_CLASSIFICATION_LOSS]
    )
    generator_losses: List[RegistrationKey[Loss]] = Param([DISCREPANCY_LOSS])


@register_class(
    name="model",
    tags={"mcd", "transformer"},
    namespace=NAMESPACE,
    component=MCD_COMPONENT,
)
class TransformerMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
