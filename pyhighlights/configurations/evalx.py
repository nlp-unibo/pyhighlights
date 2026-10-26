"""EVAL-X: the random selector, the evaluator model and its toy task."""

from typing import List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.base import SPPSelector
from pyhighlights.configurations.base import SPPShapeConfig
from pyhighlights.configurations.keys import (
    CLASSIFICATION_LOSS,
    GRU_EVALX,
    RANDOM_SELECTOR,
    TOY_EVALX_TASK,
)
from pyhighlights.configurations.tasks import ToyTaskConfig
from pyhighlights.utility.losses import Loss

EVALX_COMPONENT = "pyhighlights.components.models.spp.evalx.EvalX"


@register_class(
    key=RANDOM_SELECTOR,
    component="pyhighlights.components.models.spp.evalx.RandomSelector",
)
class RandomSelectorConfig(Configuration):
    #: ``"tokens"`` keeps each position independently, ``"spans"`` keeps one
    #: contiguous span.
    mode: str = Param("tokens")
    #: The keep rate is drawn per example, uniformly from this interval.
    min_keep: float = Param(0.0, ge=0.0, le=1.0)
    max_keep: float = Param(1.0, ge=0.0, le=1.0)


@register_class(key=GRU_EVALX, component=EVALX_COMPONENT)
class GRUEvalXConfig(SPPShapeConfig):
    name: str = Param("evalx")
    selectors: RegistrationKey[SPPSelector] = Param(RANDOM_SELECTOR)
    #: The document's label is the only target. A sparsity or contiguity term
    #: would score the random selector, which nothing can train.
    losses: List[RegistrationKey[Loss]] = Param([CLASSIFICATION_LOSS])


@register_class(
    key=TOY_EVALX_TASK,
    component="pyhighlights.components.tasks.SPPTask",
    run_method="run",
)
class ToyEvalXTaskConfig(ToyTaskConfig):
    """The synthetic corpus read by an evaluator: the end-to-end check."""

    name: str = Param("toy-evalx")
    model: RegistrationKey[Model] = Param(GRU_EVALX)
