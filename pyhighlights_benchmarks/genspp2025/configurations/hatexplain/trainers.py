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
    """The genetic search over GenSPP's generator, at the paper's settings."""

    model: RegistrationKey[GenSPP] = Param(HATEXPLAIN_GENSPP)
    #: Cross entropy above which a candidate scores the fitness floor whatever
    #: it selected, so sparsity cannot be bought with a model that has stopped
    #: classifying. Real-world text does not reach the loss a synthetic corpus
    #: does, so the bar is set where a working classifier actually sits.
    task_loss_limit: float = Param(0.6, ge=0.0)
