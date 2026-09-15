"""The paper's synthetic table: five models, one corpus."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.tasks import Task
from pyhighlights.configurations.benchmarks import BenchmarkConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.configurations.toy.keys import (
    TOY_FR_TASK,
    TOY_GENSPP_TASK,
    TOY_GRAT_TASK,
    TOY_MCD_TASK,
    TOY_MGR_TASK,
)


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
