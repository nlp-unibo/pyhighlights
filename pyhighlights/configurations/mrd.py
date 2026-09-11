"""MRD: a generator trained to make the complement stop predicting the label."""

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
    COMPLEMENT_CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    FULL_CLASSIFICATION_LOSS,
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    NAMESPACE,
    REMAINING_DISCREPANCY_LOSS,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

MRD_COMPONENT = "pyhighlights.components.models.spp.mrd.MRD"


@register_class(
    name="model", tags={"gru", "mrd"}, namespace=NAMESPACE, component=MRD_COMPONENT
)
class GRUMRDConfig(Configuration):
    name: str = Param("mrd")
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
