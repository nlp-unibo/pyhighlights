"""The paper's HateXplain table: five models, one corpus."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.tasks import Task
from pyhighlights.configurations.benchmarks import BenchmarkConfig
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_FR_TASK,
    HATEXPLAIN_GENSPP_TASK,
    HATEXPLAIN_GRAT_TASK,
    HATEXPLAIN_MCD_TASK,
    HATEXPLAIN_MGR_TASK,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="benchmark",
    tags={"hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.benchmarks.Benchmark",
    run_method="run",
)
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
