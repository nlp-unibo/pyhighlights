"""GenSPP: an SPP whose generator is searched genetically, plus its trainer."""

from typing import TYPE_CHECKING, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.backbones import (
    GRUBackboneConfig,
    TransformerBackboneConfig,
)
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    GENSPP_ADAM,
    GENSPP_GRU_BACKBONE,
    GENSPP_TRANSFORMER_BACKBONE,
    GRU_GENSPP,
    GRU_GENSPP_TRAINER,
    SIGMOID_SELECTOR,
    TOY_GENSPP_TRAINER,
    TRANSFORMER_GENSPP,
    TRANSFORMER_GENSPP_TRAINER,
)
from pyhighlights.configurations.optimizers import AdamConfig

if TYPE_CHECKING:
    import torch

    import pyhighlights.components.models.spp.base
    import pyhighlights.components.models.spp.genspp
    import pyhighlights.utility.losses

GENSPP_COMPONENT = "pyhighlights.components.models.spp.genspp.GenSPP"
GENSPP_TRAINER_COMPONENT = "pyhighlights.components.models.spp.genspp.GenSPPTrainer"


@register_class(
    key=GENSPP_GRU_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.GRUBackbone",
)
class GenSPPGRUBackboneConfig(GRUBackboneConfig):
    hidden_size: int = Param(16, ge=1)
    freeze_embeddings: bool = Param(True)
    bidirectional: bool = Param(False)
    layer_norm: bool = Param(
        False,
        description=(
            "Whether layer normalisation is applied to the recurrent states. "
            "Off, because the released GenSPP encoder passes the GRU states on "
            "unnormalised."
        ),
    )


@register_class(
    key=GENSPP_TRANSFORMER_BACKBONE,
    component="pyhighlights.components.models.spp.implementations.TransformerBackbone",
)
class GenSPPTransformerBackboneConfig(TransformerBackboneConfig):
    freeze_transformer: bool = Param(True)


@register_class(key=GENSPP_ADAM, component="torch.optim.Adam")
class GenSPPAdamConfig(AdamConfig):
    lr: float = Param(1e-2, gt=0.0)


@register_class(key=GRU_GENSPP, component=GENSPP_COMPONENT)
class GRUGenSPPConfig(SPPModelConfig):
    name: str = Param("genspp")
    selector_backbones: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(GENSPP_GRU_BACKBONE)
    selectors: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPSelector"
    ] = Param(
        SIGMOID_SELECTOR,
        description=(
            "Registration key of the selector. One sigmoid unit, as the "
            "reference generator ends, so the decision threshold is one gene."
        ),
    )
    predictor_backbone: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(GENSPP_GRU_BACKBONE)
    losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [CLASSIFICATION_LOSS]
    )
    optimizer: RegistrationKey["torch.optim.Optimizer"] = Param(GENSPP_ADAM)


@register_class(key=TRANSFORMER_GENSPP, component=GENSPP_COMPONENT)
class TransformerGenSPPConfig(GRUGenSPPConfig):
    selector_backbones: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(GENSPP_TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(GENSPP_TRANSFORMER_BACKBONE)


@register_class(key=GRU_GENSPP_TRAINER, component=GENSPP_TRAINER_COMPONENT)
class GRUGenSPPTrainerConfig(Configuration):
    model: RegistrationKey["pyhighlights.components.models.spp.genspp.GenSPP"] = Param(
        GRU_GENSPP,
        description="Registration key of the GenSPP model whose generator is searched.",
    )
    n_generations: int = Param(
        100,
        ge=0,
        description=(
            "Maximum number of generations the search breeds after scoring the "
            "first population."
        ),
    )
    population_size: int = Param(
        50,
        ge=2,
        multiple_of=2,
        description=(
            "Number of candidates each generation keeps. It has to be even, "
            "because the survivors are the best half plus a half drawn by fitness."
        ),
    )
    selection_rate: float = Param(
        0.5,
        gt=0.0,
        le=1.0,
        description=(
            "Couples per generation as a share of the population. Each couple "
            "crosses into two children, so 0.5, the reference implementation's "
            "value, adds one child per member."
        ),
    )
    mutation_probability: float = Param(
        1.0,
        gt=0.0,
        le=1.0,
        description=(
            "Probability that a mutation perturbs each gene. The default of 1.0 "
            "perturbs every gene, as the release does."
        ),
    )
    mutation_std: float = Param(
        0.05,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Standard deviation of the Gaussian noise a mutation adds to each "
            "perturbed gene."
        ),
    )
    threshold_mutation_std: float | None = Param(
        None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Standard deviation of the perturbation a mutation applies to the "
            "selector's decision threshold, or ``None`` to perturb the threshold "
            "genes at ``mutation_std`` like every other gene, as the paper does."
        ),
    )
    predictor_epochs: int = Param(
        3,
        ge=1,
        description=(
            "Number of epochs a fresh predictor trains on each candidate "
            "generator before the candidate is scored."
        ),
    )
    task_loss_limit: float = Param(
        0.1,
        ge=0.0,
        allow_inf_nan=False,
        description=(
            "Validation cross entropy above which a candidate gets the lowest "
            "fitness, 1.0, whatever it selected, so the search cannot buy a "
            "sparse selection with a predictor that has stopped classifying."
        ),
    )
    stop_threshold: float = Param(
        0.01,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "Objective, the reciprocal of the fitness, at or below which the "
            "best candidate of a generation stops the search."
        ),
    )
    seed: int | None = Param(
        None,
        description="Seed of the search's random state, or ``None`` to draw one.",
    )
    devices: List[str] = Param(
        ["cpu"],
        min_length=1,
        description=(
            "Devices that score candidates, one worker and one whole candidate "
            "at a time per device, such as ``['cpu'] * 8`` for eight cores or "
            "``['cuda:0', 'cuda:1']`` for two GPUs. CPU workers are "
            "processes and CUDA workers are threads, as ``GenSPPTrainer._score`` "
            "explains."
        ),
    )

    @classmethod
    def default(cls):
        config = super().default()
        config.add_condition(
            name="draws_a_couple",
            description=(
                "A generation draws int(selection_rate * population_size) "
                "couples, and none means a generation has no children."
            ),
            condition=lambda search: (
                int(search.selection_rate * search.population_size) >= 1
            ),
        )
        return config


@register_class(key=TRANSFORMER_GENSPP_TRAINER, component=GENSPP_TRAINER_COMPONENT)
class TransformerGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    model: RegistrationKey["pyhighlights.components.models.spp.genspp.GenSPP"] = Param(
        TRANSFORMER_GENSPP
    )


@register_class(key=TOY_GENSPP_TRAINER, component=GENSPP_TRAINER_COMPONENT)
class ToyGenSPPTrainerConfig(GRUGenSPPTrainerConfig):
    """A search small enough to finish: two candidates, one generation.

    Nothing here is a sensible experiment. It exists so the wiring of corpus,
    search, scoring and serialization can be exercised in seconds.
    """

    n_generations: int = Param(1, ge=0)
    population_size: int = Param(2, ge=2, multiple_of=2)
    predictor_epochs: int = Param(1, ge=1)
    task_loss_limit: float = Param(10.0, ge=0.0, allow_inf_nan=False)
