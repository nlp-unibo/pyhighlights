"""The penalties and the guiding terms, at the paper's coefficients."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.keys import (
    JS_DIV,
    MASKED_BCE,
)
from pyhighlights.configurations.losses import (
    GuideLossConfig,
    JSDLossConfig,
    SparsityLossConfig,
    SparsityPenaltyConfig,
)
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_SPARSITY,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)

LOSS_COMPONENT = "pyhighlights.utility.losses.Loss"


@register_class(
    name="criterion",
    tags={"hatexplain", "sparsity"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.SparsityPenalty",
)
class HateXplainSparsityConfig(SparsityPenaltyConfig):
    """The share of a post the paper asks a selector to keep."""

    threshold: float = Param(0.22, ge=0.0, le=1.0)


@register_class(
    name="loss",
    tags={"hatexplain", "sparsity"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class HateXplainSparsityLossConfig(SparsityLossConfig):
    loss: RegistrationKey = Param(HATEXPLAIN_SPARSITY)


@register_class(
    name="loss",
    tags={"guide", "hatexplain"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class HateXplainGuideLossConfig(GuideLossConfig):
    loss: RegistrationKey = Param(MASKED_BCE)
    coefficient: float = Param(2.5, ge=0.0)


@register_class(
    name="loss",
    tags={"hatexplain", "jsd"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class HateXplainJSDLossConfig(JSDLossConfig):
    loss: RegistrationKey = Param(JS_DIV)
    coefficient: float = Param(1.5, ge=0.0)
