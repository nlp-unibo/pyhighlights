"""HateXplain as the paper prepares it, which is not as it is distributed.

Three choices turn the corpus into the paper's benchmark, and all three change
the numbers:

* posts over thirty tokens are **dropped**, which is how the released code
  bounds its compute;
* ``offensive`` is folded into ``hatespeech`` **before** the annotators are
  counted, leaving a two-class task -- fold it afterwards and a post the three
  annotators split three ways gets a different label;
* the surviving votes and rationales are reduced by majority.

Tokens are embedded with GloVe ``twitter.27B`` at 25 dimensions, frozen, and
the vocabulary is restricted to what the release covers. That file is a
1.4 GB download the paper expects you to fetch yourself, so it is a path the
task is given rather than a URL it fetches::

    Registry.from_key(HATEXPLAIN_FR_TASK, embeddings="glove.twitter.27B.25d.txt")
"""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.components.tasks import Task
from pyhighlights.configurations.backbones import GRUBackboneConfig
from pyhighlights.configurations.benchmarks import BenchmarkConfig
from pyhighlights.configurations.fr import GRUFRConfig
from pyhighlights.configurations.genspp import GRUGenSPPConfig, GRUGenSPPTrainerConfig
from pyhighlights.configurations.grat import AttentionGuiderConfig, GRUGRATConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    DISCREPANCY_LOSS,
    FULL_CLASSIFICATION_LOSS,
    HATEXPLAIN,
    JS_DIV,
    MASKED_BCE,
)
from pyhighlights.configurations.losses import (
    GuideLossConfig,
    JSDLossConfig,
    SparsityLossConfig,
    SparsityPenaltyConfig,
)
from pyhighlights.configurations.mcd import GRUMCDConfig
from pyhighlights.configurations.mgr import GRUMGRConfig
from pyhighlights.configurations.preprocessors import PipelineConfig
from pyhighlights.utility.losses import Loss
from pyhighlights_benchmarks.genspp2025.configurations.common import (
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    HATEXPLAIN_AGGREGATOR,
    HATEXPLAIN_BACKBONE,
    HATEXPLAIN_FR,
    HATEXPLAIN_FR_TASK,
    HATEXPLAIN_GENSPP,
    HATEXPLAIN_GENSPP_BACKBONE,
    HATEXPLAIN_GENSPP_TASK,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_GRAT_TASK,
    HATEXPLAIN_GUIDE_LOSS,
    HATEXPLAIN_GUIDER,
    HATEXPLAIN_JSD_LOSS,
    HATEXPLAIN_LABEL_MAPPER,
    HATEXPLAIN_LENGTH_FILTER,
    HATEXPLAIN_MCD,
    HATEXPLAIN_MCD_TASK,
    HATEXPLAIN_MGR,
    HATEXPLAIN_MGR_TASK,
    HATEXPLAIN_PIPELINE,
    HATEXPLAIN_SPARSITY,
    HATEXPLAIN_SPARSITY_LOSS,
    NAMESPACE,
)

#: GloVe twitter covers what it covers; the table is replaced on load, so this
#: is a placeholder rather than a number anybody has to get right.
VOCABULARY_SIZE = 2


class LengthFilterConfig(Configuration):
    """Posts over thirty tokens are dropped, not truncated."""

    max_length: int = Param(30, ge=1)

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"length", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.LengthFilter",
    )
    def default(cls):
        return super().default()


class LabelMapperConfig(Configuration):
    """``offensive`` becomes ``hatespeech``, before the votes are counted."""

    mapping: dict = Param({"offensive": "hatespeech"})
    column: str = Param("annotator_labels")

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"labels", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.LabelMapper",
    )
    def default(cls):
        return super().default()


class AggregatorConfig(Configuration):
    """Two classes left, so three annotators always have a majority."""

    labels: List[str] = Param(["hatespeech", "normal"])
    rationale: str = Param("majority")
    ties: str = Param("drop")

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"aggregator", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.AnnotationAggregator",
    )
    def default(cls):
        return super().default()


class HateXplainPipelineConfig(PipelineConfig):
    """Filter, fold, then reduce -- in that order."""

    steps: List[RegistrationKey[Preprocessor]] = Param(
        [HATEXPLAIN_LENGTH_FILTER, HATEXPLAIN_LABEL_MAPPER, HATEXPLAIN_AGGREGATOR]
    )

    @classmethod
    @register_method(
        name="preprocessor",
        tags={"pipeline", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.preprocessors.Pipeline",
    )
    def default(cls):
        return super().default()


class HateXplainSparsityConfig(SparsityPenaltyConfig):
    """A higher selection target than the toy corpus asks for."""

    threshold: float = Param(0.22, ge=0.0, le=1.0)

    @classmethod
    @register_method(
        name="criterion",
        tags={"sparsity", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.SparsityPenalty",
    )
    def default(cls):
        return super().default()


class HateXplainSparsityLossConfig(SparsityLossConfig):
    loss: RegistrationKey = Param(HATEXPLAIN_SPARSITY)

    @classmethod
    @register_method(
        name="loss",
        tags={"sparsity", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class HateXplainGuideLossConfig(GuideLossConfig):
    loss: RegistrationKey = Param(MASKED_BCE)
    coefficient: float = Param(2.5, ge=0.0)

    @classmethod
    @register_method(
        name="loss",
        tags={"guide", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class HateXplainJSDLossConfig(JSDLossConfig):
    loss: RegistrationKey = Param(JS_DIV)
    coefficient: float = Param(1.5, ge=0.0)

    @classmethod
    @register_method(
        name="loss",
        tags={"jsd", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class HateXplainBackboneConfig(GRUBackboneConfig):
    vocab_size: int = Param(VOCABULARY_SIZE, ge=1)
    embedding_dim: int = Param(25, ge=1)
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    dropout_rate: float = Param(0.0, ge=0.0, lt=1.0)

    @classmethod
    @register_method(
        name="backbone",
        tags={"hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class HateXplainGenSPPBackboneConfig(HateXplainBackboneConfig):
    bidirectional: bool = Param(False)

    @classmethod
    @register_method(
        name="backbone",
        tags={"hatexplain", "genspp"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.implementations.GRUBackbone",
    )
    def default(cls):
        return super().default()


class HateXplainGuiderConfig(AttentionGuiderConfig):
    backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    noise_sigma: float = Param(1.0, ge=0.0)

    @classmethod
    @register_method(
        name="guider",
        tags={"hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.AttentionGuider",
    )
    def default(cls):
        return super().default()


class HateXplainFRConfig(GRUFRConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )

    @classmethod
    @register_method(
        name="model",
        tags={"fr", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.fr.FR",
    )
    def default(cls):
        return super().default()


class HateXplainMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE, HATEXPLAIN_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )

    @classmethod
    @register_method(
        name="model",
        tags={"mgr", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mgr.MGR",
    )
    def default(cls):
        return super().default()


class HateXplainMCDConfig(GRUMCDConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    rationale_losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, HATEXPLAIN_SPARSITY_LOSS]
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param([FULL_CLASSIFICATION_LOSS])
    generator_losses: List[RegistrationKey[Loss]] = Param([DISCREPANCY_LOSS])

    @classmethod
    @register_method(
        name="model",
        tags={"mcd", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.mcd.MCD",
    )
    def default(cls):
        return super().default()


class HateXplainGRATConfig(GRUGRATConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_BACKBONE)
    guider: RegistrationKey = Param(HATEXPLAIN_GUIDER)
    losses: List[RegistrationKey[Loss]] = Param(
        [
            CLASSIFICATION_LOSS,
            HATEXPLAIN_SPARSITY_LOSS,
            HATEXPLAIN_GUIDE_LOSS,
            HATEXPLAIN_JSD_LOSS,
        ]
    )
    guide_decay: float = Param(1e-5, ge=0.0)
    pretrain_epochs: int = Param(10, ge=0)

    @classmethod
    @register_method(
        name="model",
        tags={"grat", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.grat.GRAT",
    )
    def default(cls):
        return super().default()


class HateXplainGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(HATEXPLAIN_GENSPP_BACKBONE)

    @classmethod
    @register_method(
        name="model",
        tags={"genspp", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPP",
    )
    def default(cls):
        return super().default()


class HateXplainGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """A looser expected cross entropy than the toy corpus: 0.6 against 0.1."""

    model: RegistrationKey[GenSPP] = Param(HATEXPLAIN_GENSPP)
    task_loss_limit: float = Param(0.6, ge=0.0)

    @classmethod
    @register_method(
        name="trainer",
        tags={"genspp", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
    )
    def default(cls):
        return super().default()


class HateXplainTaskConfig(PaperTaskConfig):
    """One baseline over HateXplain, with the paper's preprocessing."""

    loader: RegistrationKey = Param(HATEXPLAIN)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(HATEXPLAIN_PIPELINE)
    embeddings: str | None = Param(None)
    pretrained_tokens_only: bool = Param(True)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)


class HateXplainFRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-fr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_FR)

    @classmethod
    @register_method(
        name="task",
        tags={"fr", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.SPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HateXplainMGRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mgr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MGR)

    @classmethod
    @register_method(
        name="task",
        tags={"mgr", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.SPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HateXplainMCDTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mcd")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MCD)

    @classmethod
    @register_method(
        name="task",
        tags={"mcd", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.SPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HateXplainGRATTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-grat")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_GRAT)

    @classmethod
    @register_method(
        name="task",
        tags={"grat", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.SPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HateXplainGenSPPTaskConfig(PaperGenSPPTaskConfig):
    name: str = Param("hatexplain-genspp")
    loader: RegistrationKey = Param(HATEXPLAIN)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(HATEXPLAIN_PIPELINE)
    search: RegistrationKey = Param(HATEXPLAIN_GENSPP_TRAINER)
    embeddings: str | None = Param(None)
    pretrained_tokens_only: bool = Param(True)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)

    @classmethod
    @register_method(
        name="task",
        tags={"genspp", "hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.GenSPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class HateXplainBenchmarkConfig(BenchmarkConfig):
    """The paper's real-world table: five models, one corpus."""

    name: str = Param("genspp2025-hatexplain")
    tasks: List[RegistrationKey[Task]] = Param(
        [
            HATEXPLAIN_FR_TASK,
            HATEXPLAIN_MGR_TASK,
            HATEXPLAIN_MCD_TASK,
            HATEXPLAIN_GRAT_TASK,
            HATEXPLAIN_GENSPP_TASK,
        ]
    )

    @classmethod
    @register_method(
        name="benchmark",
        tags={"hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.benchmarks.Benchmark",
        run_method="run",
    )
    def default(cls):
        return super().default()


__all__ = ["VOCABULARY_SIZE"]
