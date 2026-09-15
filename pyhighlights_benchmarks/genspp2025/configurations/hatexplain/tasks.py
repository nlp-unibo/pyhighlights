"""One task per architecture: what a run of this corpus is."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.base import Model
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.keys import (
    HATEXPLAIN,
)
from pyhighlights_benchmarks.genspp2025.configurations.common import (
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
)
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain import VOCABULARY_SIZE
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_FR,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_MCD,
    HATEXPLAIN_MGR,
    HATEXPLAIN_PIPELINE,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)

SPP_TASK_COMPONENT = "pyhighlights.components.tasks.SPPTask"


class HateXplainTaskConfig(PaperTaskConfig):
    """One baseline over HateXplain, with the paper's preprocessing."""

    loader: RegistrationKey = Param(HATEXPLAIN)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(HATEXPLAIN_PIPELINE)
    embeddings: str | None = Param(None)
    pretrained_tokens_only: bool = Param(True)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)


@register_class(
    name="task",
    tags={"fr", "hatexplain"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainFRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-fr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_FR)


@register_class(
    name="task",
    tags={"hatexplain", "mgr"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainMGRTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mgr")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MGR)


@register_class(
    name="task",
    tags={"hatexplain", "mcd"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainMCDTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-mcd")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_MCD)


@register_class(
    name="task",
    tags={"grat", "hatexplain"},
    namespace=NAMESPACE,
    component=SPP_TASK_COMPONENT,
    run_method="run",
)
class HateXplainGRATTaskConfig(HateXplainTaskConfig):
    name: str = Param("hatexplain-grat")
    model: RegistrationKey[Model] = Param(HATEXPLAIN_GRAT)


@register_class(
    name="task",
    tags={"genspp", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class HateXplainGenSPPTaskConfig(PaperGenSPPTaskConfig):
    name: str = Param("hatexplain-genspp")
    loader: RegistrationKey = Param(HATEXPLAIN)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(HATEXPLAIN_PIPELINE)
    search: RegistrationKey = Param(HATEXPLAIN_GENSPP_TRAINER)
    embeddings: str | None = Param(None)
    pretrained_tokens_only: bool = Param(True)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)
