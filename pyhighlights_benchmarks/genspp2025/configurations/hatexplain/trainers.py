"""The genetic search over GenSPP's generator."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_GENSPP,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="trainer",
    tags={"genspp", "hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.models.spp.genspp.GenSPPTrainer",
)
class HateXplainGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """A looser expected cross entropy than the toy corpus: 0.6 against 0.1."""

    model: RegistrationKey[GenSPP] = Param(HATEXPLAIN_GENSPP)
    task_loss_limit: float = Param(0.6, ge=0.0)
