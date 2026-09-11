"""MRD: a generator trained to make the complement stop predicting the label."""

from typing import List

import torch as th
from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
)
from pyhighlights.configurations.base import PhasedSPPModelConfig
from pyhighlights.configurations.keys import (
    ADAM,
    COMPLEMENT_CLASSIFICATION_LOSS,
    FULL_CLASSIFICATION_LOSS,
    GRU_BACKBONE,
    NAMESPACE,
    REMAINING_DISCREPANCY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

MRD_COMPONENT = "pyhighlights.components.models.spp.mrd.MRD"


@register_class(
    name="model", tags={"gru", "mrd"}, namespace=NAMESPACE, component=MRD_COMPONENT
)
class GRUMRDConfig(PhasedSPPModelConfig):
    name: str = Param("mrd")
    #: The predictor reads the complement and the full input with its own
    #: encoder, so its backbone is required rather than optional.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    #: The predictor reads the complement and the full input, and never the
    #: highlight: MRD's criterion is what the remainder can still say.
    predictor_losses: List[RegistrationKey[Loss]] = Param(
        [COMPLEMENT_CLASSIFICATION_LOSS, FULL_CLASSIFICATION_LOSS]
    )
    generator_losses: List[RegistrationKey[Loss]] = Param([REMAINING_DISCREPANCY_LOSS])
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    test_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)


@register_class(
    name="model",
    tags={"mrd", "transformer"},
    namespace=NAMESPACE,
    component=MRD_COMPONENT,
)
class TransformerMRDConfig(GRUMRDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
