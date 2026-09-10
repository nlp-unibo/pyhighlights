"""MGR: several independent generators feeding one shared predictor."""

from typing import List, Literal

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
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

MGR_COMPONENT = "pyhighlights.components.models.spp.mgr.MGR"


@register_class(
    name="model", tags={"gru", "mgr"}, namespace=NAMESPACE, component=MGR_COMPONENT
)
class GRUMGRConfig(Configuration):
    name: str = Param("mgr")
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [GRU_BACKBONE, GRU_BACKBONE, GRU_BACKBONE]
    )
    selectors: List[RegistrationKey[SPPSelector]] = Param(
        [MLP_SELECTOR, MLP_SELECTOR, MLP_SELECTOR]
    )
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    select_over: str = Param("word")
    inference_head: int = Param(0, ge=0)
    loss_reduction: Literal["sum", "mean"] = Param("sum")
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    test_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)

    @classmethod
    def default(cls):
        config = super().default()
        # One backbone per generator, at least two generators, and a head that
        # exists. Declared rather than checked in the component: the registry
        # validates conditions while it expands keys, so a grid over the
        # generator count drops the impossible combinations before anything
        # trains, instead of raising at the first batch of the run that reaches
        # one.
        config.add_condition(
            name="one_backbone_per_generator",
            description="MGR pairs every generator with its own backbone.",
            condition=lambda model: (
                len(model.selector_backbones) == len(model.selectors)
            ),
        )
        config.add_condition(
            name="at_least_two_generators",
            description="MGR disagrees between generators, so it needs two of them.",
            condition=lambda model: len(model.selectors) >= 2,
        )
        config.add_condition(
            name="inference_head_exists",
            description="The head that predicts at inference is one of the generators.",
            condition=lambda model: model.inference_head < len(model.selectors),
        )
        return config


@register_class(
    name="model",
    tags={"mgr", "transformer"},
    namespace=NAMESPACE,
    component=MGR_COMPONENT,
)
class TransformerMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
