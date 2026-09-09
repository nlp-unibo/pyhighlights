"""GenSPP: an SPP whose generator is searched genetically, plus its trainer."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.backbones import (
    GRUBackboneConfig,
    TransformerBackboneConfig,
)
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    GENSPP_ADAM,
    GENSPP_GRU_BACKBONE,
    GENSPP_TRANSFORMER_BACKBONE,
    GRU_GENSPP,
    NAMESPACE,
    TRANSFORMER_GENSPP,
)
from pyhighlights.configurations.optimizers import AdamConfig
from pyhighlights.utility.losses import Loss

GENSPP_COMPONENT = "pyhighlights.components.models.spp.genspp.GenSPP"
GENSPP_TRAINER_COMPONENT = "pyhighlights.components.models.spp.genspp.GenSPPTrainer"


@register_class(
    name="backbone",
    tags={"genspp", "gru"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.GRUBackbone",
)
class GenSPPGRUBackboneConfig(GRUBackboneConfig):
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    bidirectional: bool = Param(False)


@register_class(
    name="backbone",
    tags={"genspp", "transformer"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
)
class GenSPPTransformerBackboneConfig(TransformerBackboneConfig):
    freeze_transformer: bool = Param(True)


@register_class(
    name="optimizer",
    tags={"adam", "genspp"},
    namespace=NAMESPACE,
    component="torch.optim.Adam",
)
class GenSPPAdamConfig(AdamConfig):
    lr: float = Param(1e-2, gt=0.0)


@register_class(
    name="model",
    tags={"genspp", "gru"},
    namespace=NAMESPACE,
    component=GENSPP_COMPONENT,
)
class GRUGenSPPConfig(SPPModelConfig):
    name: str = Param("genspp")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GENSPP_GRU_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GENSPP_GRU_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param([CLASSIFICATION_LOSS])
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(GENSPP_ADAM)


@register_class(
    name="model",
    tags={"genspp", "transformer"},
    namespace=NAMESPACE,
    component=GENSPP_COMPONENT,
)
class TransformerGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(
        GENSPP_TRANSFORMER_BACKBONE
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        GENSPP_TRANSFORMER_BACKBONE
    )


@register_class(
    name="trainer",
    tags={"genspp", "gru"},
    namespace=NAMESPACE,
    component=GENSPP_TRAINER_COMPONENT,
)
class GRUGenSPPTrainerConfig(Configuration):
    model: RegistrationKey[GenSPP] = Param(GRU_GENSPP)
    n_generations: int = Param(100, ge=0)
    population_size: int = Param(50, ge=2)
    mutation_probability: float = Param(1.0, gt=0.0, le=1.0)
    mutation_std: float = Param(0.05, gt=0.0)
    predictor_epochs: int = Param(3, ge=1)
    task_loss_limit: float = Param(0.1, ge=0.0)
    stop_threshold: float = Param(0.01, gt=0.0)
    seed: int | None = Param(None)
    device: str = Param("cpu")


@register_class(
    name="trainer",
    tags={"genspp", "transformer"},
    namespace=NAMESPACE,
    component=GENSPP_TRAINER_COMPONENT,
)
class TransformerGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    model: RegistrationKey[GenSPP] = Param(TRANSFORMER_GENSPP)


@register_class(
    name="trainer",
    tags={"genspp", "gru", "toy"},
    namespace=NAMESPACE,
    component=GENSPP_TRAINER_COMPONENT,
)
class ToyGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """A search small enough to finish: two candidates, one generation.

    Nothing here is a sensible experiment. It exists so the wiring -- corpus,
    search, scoring, serialization -- can be exercised in seconds.
    """

    n_generations: int = Param(1, ge=0)
    population_size: int = Param(2, ge=2)
    predictor_epochs: int = Param(1, ge=1)
    task_loss_limit: float = Param(10.0, ge=0.0)
