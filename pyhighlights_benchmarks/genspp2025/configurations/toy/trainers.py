"""The genetic search over GenSPP's generator."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.genspp import GenSPP
from pyhighlights.configurations.genspp import GRUGenSPPTrainerConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.configurations.toy.keys import (
    TOY_GENSPP,
)


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
