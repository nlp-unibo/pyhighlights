"""MGR: several independent generators feeding one shared predictor."""

from typing import List, Literal

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
    SPPSelector,
)
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    GRU_BACKBONE,
    GRU_MGR,
    MLP_SELECTOR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_MGR,
    UNNORMALISED_GRU_BACKBONE,
)

MGR_COMPONENT = "pyhighlights.components.models.spp.mgr.MGR"


@register_class(key=GRU_MGR, component=MGR_COMPONENT)
class GRUMGRConfig(SPPModelConfig):
    name: str = Param("mgr")
    #: As the original implementation: a sampled selection at evaluation and
    #: no empty-selection repair.
    sample_at_evaluation: bool = Param(True)
    repair_empty_selections: bool = Param(False)
    #: One backbone and one selector *per generator*, where every other
    #: architecture has one of each. That is the whole of MGR, so the base
    #: class cannot carry these as defaults.
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [GRU_BACKBONE, GRU_BACKBONE, GRU_BACKBONE]
    )
    selectors: List[RegistrationKey[SPPSelector]] = Param(
        [MLP_SELECTOR, MLP_SELECTOR, MLP_SELECTOR]
    )
    #: The generators share one predictor, and it encodes with its own
    #: backbone, so this is required rather than optional.
    #: Unnormalised, as the original implementation's predictor encoder.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(UNNORMALISED_GRU_BACKBONE)
    inference_head: int = Param(0, ge=0)
    #: How the heads' losses combine. ``"mean"`` divides them by the number of
    #: heads, which rescales the logged loss. Under Adam it barely changes
    #: training, since an Adam update does not depend on the loss scale.
    loss_reduction: Literal["sum", "mean"] = Param("sum")

    @classmethod
    def default(cls):
        config = super().default()
        # One backbone per generator, at least two generators, and a head that
        # exists. Declared rather than checked in the component: the registry
        # validates conditions while it expands keys, so a grid over the
        # generator count drops the impossible combinations before anything
        # trains, instead of raising at the first batch of the run that reaches
        # one.
        config.add_condition(
            name="one_backbone_per_generator",
            description="MGR pairs every generator with its own backbone.",
            condition=lambda model: (
                len(model.selector_backbones) == len(model.selectors)
            ),
        )
        config.add_condition(
            name="at_least_two_generators",
            description="MGR disagrees between generators, so it needs two of them.",
            condition=lambda model: len(model.selectors) >= 2,
        )
        config.add_condition(
            name="inference_head_exists",
            description="The head that predicts at inference is one of the generators.",
            condition=lambda model: model.inference_head < len(model.selectors),
        )
        return config


@register_class(key=TRANSFORMER_MGR, component=MGR_COMPONENT)
class TransformerMGRConfig(GRUMGRConfig):
    selector_backbones: List[RegistrationKey[SPPBackbone]] = Param(
        [TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE, TRANSFORMER_BACKBONE]
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
