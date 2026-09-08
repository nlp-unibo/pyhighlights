"""MGR: several independent generators feeding one shared predictor."""

from typing import List, Literal

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

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
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

#: Namespace cinnamon resolves this module's registrations under. Kept a literal
#: in every registering module: ``NamespaceExtractor`` reads it statically and only
#: sees bindings made in the same file.
NAMESPACE = "pyhighlights"


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
    @register_method(
        name="model",
        tags={"mgr", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mgr.MGR",
    )
    def default(cls):
        return super().default()


class TransformerMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)

    @classmethod
    @register_method(
        name="model",
        tags={"mgr", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mgr.MGR",
    )
    def default(cls):
        return super().default()
