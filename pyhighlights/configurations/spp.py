from typing import Dict, List, Literal

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method
from torchmetrics import Metric

from pyhighlights.components.models.spp.base import (
    SPPAggregator,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.components.models.spp.grat import GRATGuider
from pyhighlights.utility.losses import Loss

NAMESPACE = "pyhighlights"


def key(name: str, *tags: str) -> RegistrationKey:
    return RegistrationKey(name=name, tags=set(tags), namespace=NAMESPACE)


GRU_BACKBONE = key("backbone", "gru")
TRANSFORMER_BACKBONE = key("backbone", "transformer")
MLP_SELECTOR = key("selector", "mlp")
MLP_PREDICTOR = key("predictor", "mlp")
GRU_GUIDER = key("guider", "attention", "gru")
TRANSFORMER_GUIDER = key("guider", "attention", "transformer")
CROSS_ENTROPY = key("criterion", "cross_entropy")
CLASSIFICATION_LOSS = key("loss", "classification")
SPARSITY_LOSS = key("loss", "sparsity")
CONTIGUITY_LOSS = key("loss", "contiguity")
ADAM = key("optimizer", "adam")
GRU_FR = key("model", "fr", "gru")
GRU_MGR = key("model", "mgr", "gru")
GRU_MCD = key("model", "mcd", "gru")
GRU_GRAT = key("model", "grat", "gru")
TRANSFORMER_FR = key("model", "fr", "transformer")
TRANSFORMER_MGR = key("model", "mgr", "transformer")
TRANSFORMER_MCD = key("model", "mcd", "transformer")
TRANSFORMER_GRAT = key("model", "grat", "transformer")


class GRUBackboneConfig(Configuration):
    vocab_size: int = Param(10_000, ge=1)
    embedding_dim: int = Param(128, ge=1)
    hidden_size: int = Param(128, ge=1)
    freeze_embeddings: bool = Param(False)
    num_layers: int = Param(1, ge=1)
    bidirectional: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)

    @classmethod
    @register_method(
        name="backbone",
        tags={"gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class TransformerBackboneConfig(Configuration):
    pretrained_model_card: str = Param("distilbert-base-uncased")
    num_features: int | None = Param(None, ge=1)
    freeze_transformer: bool = Param(False)

    @classmethod
    @register_method(
        name="backbone",
        tags={"transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
    )
    def default(cls):
        return super().default()


class MLPSelectorConfig(Configuration):
    hidden_sizes: List[int] = Param([])

    @classmethod
    @register_method(
        name="selector",
        tags={"mlp"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.MLPSelector",
    )
    def default(cls):
        return super().default()


class MLPPredictorConfig(Configuration):
    hidden_sizes: List[int] = Param([])
    num_classes: int = Param(2, ge=2)

    @classmethod
    @register_method(
        name="predictor",
        tags={"mlp"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.MLPPredictor",
    )
    def default(cls):
        return super().default()


class AttentionGuiderConfig(Configuration):
    backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    noise_sigma: float = Param(1.0, ge=0.0)

    @classmethod
    @register_method(
        name="guider",
        tags={"attention", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.AttentionGuider",
    )
    def default(cls):
        return super().default()


class TransformerAttentionGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)

    @classmethod
    @register_method(
        name="guider",
        tags={"attention", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.AttentionGuider",
    )
    def default(cls):
        return super().default()


class CrossEntropyConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"cross_entropy"},
        namespace=NAMESPACE,
        component="torch.nn.CrossEntropyLoss",
    )
    def default(cls):
        return super().default()


class ClassificationLossConfig(Configuration):
    name: str = Param("classification")
    loss: RegistrationKey[th.nn.CrossEntropyLoss] = Param(CROSS_ENTROPY)
    coefficient: float = Param(1.0, ge=0.0)
    enabled: bool = Param(True)

    @classmethod
    @register_method(
        name="loss",
        tags={"classification"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.ClassificationLoss",
    )
    def default(cls):
        return super().default()


class SparsityLossConfig(Configuration):
    name: str = Param("sparsity")
    coefficient: float = Param(1.0, ge=0.0)
    enabled: bool = Param(True)
    sparsity_threshold: float = Param(0.15, ge=0.0, le=1.0)

    @classmethod
    @register_method(
        name="loss",
        tags={"sparsity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.HighlightSparsityLoss",
    )
    def default(cls):
        return super().default()


class ContiguityLossConfig(Configuration):
    name: str = Param("contiguity")
    coefficient: float = Param(2.0, ge=0.0)
    enabled: bool = Param(True)

    @classmethod
    @register_method(
        name="loss",
        tags={"contiguity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.HighlightContiguityLoss",
    )
    def default(cls):
        return super().default()


class AdamConfig(Configuration):
    lr: float = Param(1e-3, gt=0.0)
    weight_decay: float = Param(0.0, ge=0.0)

    @classmethod
    @register_method(
        name="optimizer",
        tags={"adam"},
        namespace=NAMESPACE,
        component="torch.optim.Adam",
    )
    def default(cls):
        return super().default()


class GRUFRConfig(Configuration):
    name: str = Param("fr")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] | None = Param(None)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    val_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    test_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)

    @classmethod
    @register_method(
        name="model",
        tags={"fr", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.fr.FR",
    )
    def default(cls):
        return super().default()


class GRUGRATConfig(Configuration):
    name: str = Param("grat")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    guider: RegistrationKey[GRATGuider] = Param(GRU_GUIDER)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    classification_loss: RegistrationKey[Loss] = Param(CLASSIFICATION_LOSS)
    rationale_losses: List[RegistrationKey[Loss]] = Param(
        [SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    pretrain_epochs: int = Param(10, ge=0)
    guide_coefficient: float = Param(1.0, ge=0.0)
    jsd_coefficient: float = Param(1.0, ge=0.0)
    guide_decay: float = Param(1e-4, ge=0.0)
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    val_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    test_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)

    @classmethod
    @register_method(
        name="model",
        tags={"grat", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.GRAT",
    )
    def default(cls):
        return super().default()


class GRUMCDConfig(Configuration):
    name: str = Param("mcd")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    classification_loss: RegistrationKey[Loss] = Param(CLASSIFICATION_LOSS)
    rationale_losses: List[RegistrationKey[Loss]] = Param(
        [SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    discrepancy_coefficient: float = Param(1.0, ge=0.0)
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    val_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    test_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)

    @classmethod
    @register_method(
        name="model",
        tags={"mcd", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mcd.MCD",
    )
    def default(cls):
        return super().default()


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
    train_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    val_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)
    test_metrics: Dict[str, RegistrationKey[Metric]] | None = Param(None)

    @classmethod
    @register_method(
        name="model",
        tags={"mgr", "gru"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mgr.MGR",
    )
    def default(cls):
        return super().default()


class TransformerFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)

    @classmethod
    @register_method(
        name="model",
        tags={"fr", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.fr.FR",
    )
    def default(cls):
        return super().default()


class TransformerGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    guider: RegistrationKey[GRATGuider] = Param(TRANSFORMER_GUIDER)

    @classmethod
    @register_method(
        name="model",
        tags={"grat", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.GRAT",
    )
    def default(cls):
        return super().default()


class TransformerMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)

    @classmethod
    @register_method(
        name="model",
        tags={"mcd", "transformer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mcd.MCD",
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
