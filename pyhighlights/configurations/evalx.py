"""EVAL-X: the random selector, the evaluator model and its toy task."""

from typing import TYPE_CHECKING, List, Literal

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.base import SPPShapeConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    GRU_EVALX,
    RANDOM_SELECTOR,
    TOY_EVALX_TASK,
)
from pyhighlights.configurations.tasks import ToyTaskConfig

if TYPE_CHECKING:
    import pyhighlights.components.models.base
    import pyhighlights.components.models.spp.base
    import pyhighlights.utility.losses

EVALX_COMPONENT = "pyhighlights.components.models.spp.evalx.EvalX"


@register_class(
    key=RANDOM_SELECTOR,
    component="pyhighlights.components.models.spp.evalx.RandomSelector",
)
class RandomSelectorConfig(Configuration):
    mode: Literal["tokens", "spans"] = Param(
        "tokens",
        description=(
            "How the selector keeps positions: ``'tokens'`` keeps each "
            "position independently, and ``'spans'`` keeps one contiguous span."
        ),
    )
    min_keep: float = Param(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Lower bound of the keep rate, which is drawn per example, uniformly "
            "from ``[min_keep, max_keep]``."
        ),
    )
    max_keep: float = Param(
        1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Upper bound of the keep rate, which is drawn per example, uniformly "
            "from ``[min_keep, max_keep]``."
        ),
    )

    @classmethod
    def default(cls):
        config = super().default()
        config.add_condition(
            name="ordered_keep_rates",
            description="The keep rate is drawn from [min_keep, max_keep].",
            condition=lambda selector: selector.min_keep <= selector.max_keep,
        )
        return config


@register_class(key=GRU_EVALX, component=EVALX_COMPONENT)
class GRUEvalXConfig(SPPShapeConfig):
    name: str = Param("evalx")
    selectors: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPSelector"
    ] = Param(RANDOM_SELECTOR)
    losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [CLASSIFICATION_LOSS],
        description=(
            "Registration keys of the losses, where the document's label is the "
            "only target. A sparsity or contiguity term would score the random "
            "selector, which nothing can train."
        ),
    )


@register_class(
    key=TOY_EVALX_TASK,
    component="pyhighlights.components.tasks.SPPTask",
    run_method="run",
)
class ToyEvalXTaskConfig(ToyTaskConfig):
    """The synthetic corpus read by an evaluator: the end-to-end check."""

    name: str = Param("toy-evalx")
    model: RegistrationKey["pyhighlights.components.models.base.Model"] = Param(
        GRU_EVALX
    )
