"""MCD: selected-input and full-input predictions trained in two phases."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPAggregator,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.configurations.keys import (
    ADAM,
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    DISCREPANCY_LOSS,
    FULL_CLASSIFICATION_LOSS,
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

MCD_COMPONENT = "pyhighlights.components.models.spp.mcd.MCD"


@register_class(
    name="model", tags={"gru", "mcd"}, namespace=NAMESPACE, component=MCD_COMPONENT
)
class GRUMCDConfig(Configuration):
    name: str = Param("mcd")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    select_over: str = Param("word")
    #: One rate for the encoders, another for everything above them. ``None``
    #: trains the whole model at the optimizer's own rate, which is what every
    #: published implementation of these architectures does -- they encode
    #: with a GRU over a frozen table, so nothing pretrained is fine-tuned.
    #: Set it when a pretrained encoder *is* being fine-tuned: one rate cannot
    #: serve both a transformer and a selector initialized from scratch.
    encoder_lr: float | None = Param(None, gt=0.0)
    rationale_losses: List[RegistrationKey[Loss]] = Param(
        [SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, FULL_CLASSIFICATION_LOSS]
    )
    generator_losses: List[RegistrationKey[Loss]] = Param([DISCREPANCY_LOSS])
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    test_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)


@register_class(
    name="model",
    tags={"mcd", "transformer"},
    namespace=NAMESPACE,
    component=MCD_COMPONENT,
)
class TransformerMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
