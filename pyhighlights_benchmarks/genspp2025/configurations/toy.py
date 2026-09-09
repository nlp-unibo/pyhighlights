"""The synthetic corpus: three hidden patterns over a twenty-character string.

Tokens are characters, so the vocabulary is the alphabet and the embedding
table is 27 rows -- twenty-six letters and the unknown id. The released
baselines freeze that table without pretraining it, which makes it a fixed
random projection rather than something the model can learn to lean on.
"""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.base import SPPBackbone, SPPPredictor
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.components.tasks import Task
from pyhighlights.configurations.backbones import GRUBackboneConfig, MLPPredictorConfig
from pyhighlights.configurations.benchmarks import BenchmarkConfig
from pyhighlights.configurations.datasets import LoaderConfig
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig, GRUGenSPPTrainerConfig
from pyhighlights.configurations.grat import AttentionGuiderConfig, GRUGRATConfig
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights_benchmarks.genspp2025.configurations.common import (
    THREE_CLASS_METRICS,
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
    TOY,
    TOY_BACKBONE,
    TOY_FR,
    TOY_FR_TASK,
    TOY_GENSPP,
    TOY_GENSPP_BACKBONE,
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
    TOY_GRAT,
    TOY_GRAT_TASK,
    TOY_GUIDER,
    TOY_MCD,
    TOY_MCD_TASK,
    TOY_MGR,
    TOY_MGR_TASK,
    TOY_PREDICTOR,
)

#: Twenty-six letters plus the unknown id.
VOCABULARY_SIZE = 27


GRU_BACKBONE_COMPONENT = (
    "pyhighlights.components.models.spp.implementations.GRUBackbone"
)
SPP_TASK_COMPONENT = "pyhighlights.components.tasks.SPPTask"


@register_class(
    name="dataset",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights_benchmarks.genspp2025.corpora.GenSPPToyLoader",
)
class ToyConfig(LoaderConfig):
    """The released ``toy_dataset.pkl``, not a regenerated corpus."""

    url: str | None = Param(None)
    sha256: str | None = Param(None)
    train_ratio: float = Param(0.8, gt=0.0, lt=1.0)
    val_ratio: float = Param(0.2, ge=0.0, lt=1.0)
    split_seed: int = Param(15000)


@register_class(
    name="backbone", tags={"toy"}, namespace=NAMESPACE, component=GRU_BACKBONE_COMPONENT
)
class ToyBackboneConfig(GRUBackboneConfig):
    vocab_size: int = Param(VOCABULARY_SIZE, ge=1)
    embedding_dim: int = Param(25, ge=1)
    hidden_size: int = Param(8, ge=1)
    freeze_embeddings: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)


@register_class(
    name="backbone",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component=GRU_BACKBONE_COMPONENT,
)
class ToyGenSPPBackboneConfig(ToyBackboneConfig):
    """The genetic half's encoder: one direction, one row per letter."""

    embedding_dim: int = Param(26, ge=1)
    bidirectional: bool = Param(False)


@register_class(
    name="predictor",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.implementations.MLPPredictor",
)
class ToyPredictorConfig(MLPPredictorConfig):
    num_classes: int = Param(3, ge=2)


@register_class(
    name="guider",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.AttentionGuider",
)
class ToyGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    noise_sigma: float = Param(1.0, ge=0.0)


@register_class(
    name="model",
    tags={"fr", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.fr.FR",
)
class ToyFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"mgr", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mgr.MGR",
)
class ToyMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TOY_BACKBONE, TOY_BACKBONE, TOY_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"mcd", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.mcd.MCD",
)
class ToyMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="model",
    tags={"grat", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.grat.GRAT",
)
class ToyGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)
    guider: RegistrationKey = Param(TOY_GUIDER)
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)


@register_class(
    name="model",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPP",
)
class ToyGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TOY_GENSPP_BACKBONE)
    predictor: RegistrationKey[SPPPredictor] = Param(TOY_PREDICTOR)


@register_class(
    name="trainer",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class ToyGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """The search as released: the expected cross entropy is 0.1 here."""

    model: RegistrationKey[GenSPP] = Param(TOY_GENSPP)
    task_loss_limit: float = Param(0.1, ge=0.0)


class ToyTaskConfig(PaperTaskConfig):
    """One baseline over the toy corpus."""

    loader: RegistrationKey = Param(TOY)
    val_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    test_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)


@register_class(
    name="task",
    tags={"fr", "toy"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class ToyFRTaskConfig(ToyTaskConfig):
    name: str = Param("toy-fr")
    model: RegistrationKey[Model] = Param(TOY_FR)


@register_class(
    name="task",
    tags={"mgr", "toy"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class ToyMGRTaskConfig(ToyTaskConfig):
    name: str = Param("toy-mgr")
    model: RegistrationKey[Model] = Param(TOY_MGR)


@register_class(
    name="task",
    tags={"mcd", "toy"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class ToyMCDTaskConfig(ToyTaskConfig):
    name: str = Param("toy-mcd")
    model: RegistrationKey[Model] = Param(TOY_MCD)


@register_class(
    name="task",
    tags={"grat", "toy"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class ToyGRATTaskConfig(ToyTaskConfig):
    name: str = Param("toy-grat")
    model: RegistrationKey[Model] = Param(TOY_GRAT)


@register_class(
    name="task",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class ToyGenSPPTaskConfig(PaperGenSPPTaskConfig):
    name: str = Param("toy-genspp")
    loader: RegistrationKey = Param(TOY)
    search: RegistrationKey = Param(TOY_GENSPP_TRAINER)
    val_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    test_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)


@register_class(
    name="benchmark",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.benchmarks.Benchmark",
    run_method="run",
)
class ToyBenchmarkConfig(BenchmarkConfig):
    """The paper's synthetic table: five models, one corpus."""

    name: str = Param("genspp2025-toy")
    tasks: List[RegistrationKey[Task]] = Param(
        [TOY_FR_TASK, TOY_MGR_TASK, TOY_MCD_TASK, TOY_GRAT_TASK, TOY_GENSPP_TASK]
    )


__all__ = ["VOCABULARY_SIZE"]
