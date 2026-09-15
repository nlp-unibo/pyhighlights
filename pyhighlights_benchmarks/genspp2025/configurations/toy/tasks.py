"""One task per architecture: what a run of this corpus is."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.base import Model
from pyhighlights_benchmarks.genspp2025.configurations.common import (
    THREE_CLASS_METRICS,
    PaperGenSPPTaskConfig,
    PaperTaskConfig,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.configurations.toy import VOCABULARY_SIZE
from pyhighlights_benchmarks.genspp2025.configurations.toy.keys import (
    TOY,
    TOY_FR,
    TOY_GENSPP_TRAINER,
    TOY_GRAT,
    TOY_MCD,
    TOY_MGR,
)

SPP_TASK_COMPONENT = "pyhighlights.components.tasks.SPPTask"


class ToyTaskConfig(PaperTaskConfig):
    """One baseline over the toy corpus."""

    loader: RegistrationKey = Param(TOY)
    val_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    test_metrics: List[RegistrationKey] = Param(THREE_CLASS_METRICS)
    vocabulary_size: int = Param(VOCABULARY_SIZE, ge=2)
    #: One-hot inputs, as the release reads this corpus. The width is the
    #: vocabulary's own size here, and has to match the backbone's
    #: ``embedding_dim``.
    one_hot_embeddings: int = Param(25, ge=1)


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
    #: Twenty-six, which is the width the genetic half declares -- two
    #: columns wider than the corpus has characters.
    one_hot_embeddings: int = Param(26, ge=1)
