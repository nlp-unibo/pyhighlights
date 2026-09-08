"""GenSPP: an SPP whose generator is searched genetically, plus its trainer."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

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


class GenSPPGRUBackboneConfig(GRUBackboneConfig):
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    bidirectional: bool = Param(False)

    @classmethod
    @register_method(
        name="backbone",
        tags={"genspp", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class GenSPPTransformerBackboneConfig(TransformerBackboneConfig):
    freeze_transformer: bool = Param(True)

    @classmethod
    @register_method(
        name="backbone",
        tags={"genspp", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
    )
    def default(cls):
        return super().default()


class GenSPPAdamConfig(AdamConfig):
    lr: float = Param(1e-2, gt=0.0)

    @classmethod
    @register_method(
        name="optimizer",
        tags={"adam", "genspp"},
        namespace=NAMESPACE,
        component="torch.optim.Adam",
    )
    def default(cls):
        return super().default()


class GRUGenSPPConfig(SPPModelConfig):
    name: str = Param("genspp")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GENSPP_GRU_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GENSPP_GRU_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param([CLASSIFICATION_LOSS])
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(GENSPP_ADAM)

    @classmethod
    @register_method(
        name="model",
        tags={"genspp", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPP",
    )
    def default(cls):
        return super().default()


class TransformerGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(
        GENSPP_TRANSFORMER_BACKBONE
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        GENSPP_TRANSFORMER_BACKBONE
    )

    @classmethod
    @register_method(
        name="model",
        tags={"genspp", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPP",
    )
    def default(cls):
        return super().default()


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

    @classmethod
    @register_method(
        name="trainer",
        tags={"genspp", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
    )
    def default(cls):
        return super().default()


class TransformerGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    model: RegistrationKey[GenSPP] = Param(TRANSFORMER_GENSPP)

    @classmethod
    @register_method(
        name="trainer",
        tags={"genspp", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
    )
    def default(cls):
        return super().default()
