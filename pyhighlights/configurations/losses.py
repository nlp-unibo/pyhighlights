"""Criterion registrations and the bindings that feed them named fields."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.keys import (
    CONTIGUITY_PENALTY,
    CROSS_ENTROPY,
    JS_DIV,
    KL_DIV,
    KNOWLEDGE_BCE,
    MASKED_BCE,
    MASKED_CROSS_ENTROPY,
    NAMESPACE,
    SPARSITY_PENALTY,
)

LOSS_COMPONENT = "pyhighlights.utility.losses.Loss"


@register_class(
    name="criterion",
    tags={"cross_entropy"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.CrossEntropy",
)
class CrossEntropyConfig(Configuration):
    """Cross entropy, weighted per class where a corpus needs it."""

    #: One weight per class, or nothing for an unweighted loss. A corpus with
    #: 106 positives in 20,417 sentences is answered correctly by a model that
    #: never predicts one, so the numbers a class-imbalanced corpus needs are
    #: part of its configuration rather than a detail of its training.
    weight: List[float] | None = Param(None)


@register_class(
    name="criterion",
    tags={"masked_cross_entropy"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.MaskedCrossEntropy",
)
class MaskedCrossEntropyConfig(Configuration):
    """Cross entropy over valid, labelled positions of any axis."""

    #: One weight per class, for an axis whose classes are imbalanced. The
    #: knowledge axis is: a clause instantiates one or two of up to 28
    #: rationales, before the clauses that instantiate none are counted.
    weight: List[float] | None = Param(None)


@register_class(
    name="criterion",
    tags={"masked_bce", "knowledge"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.MaskedBinaryCrossEntropy",
)
class KnowledgeBCEConfig(Configuration):
    """A binary criterion carrying one positive weight per knowledge entry."""

    ignore_index: int = Param(-1)
    pos_weight: List[float] | None = Param(None)


@register_class(
    name="criterion",
    tags={"masked_bce"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.MaskedBinaryCrossEntropy",
)
class MaskedBCEConfig(Configuration):
    """One independent decision per position of the axis being scored."""

    ignore_index: int = Param(-1)
    #: One factor per position, multiplying the cost of missing a positive
    #: there. Read off the corpus by ``KnowledgeWeights`` rather than typed.
    pos_weight: List[float] | None = Param(None)


@register_class(
    name="criterion",
    tags={"kl_div"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.KLDiv",
)
class KLDivConfig(Configuration):
    pass


@register_class(
    name="criterion",
    tags={"js_div"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.JSDiv",
)
class JSDivConfig(Configuration):
    pass


@register_class(
    name="criterion",
    tags={"contiguity"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.ContiguityPenalty",
)
class ContiguityPenaltyConfig(Configuration):
    pass


@register_class(
    name="criterion",
    tags={"sparsity"},
    namespace=NAMESPACE,
    component="pyhighlights.utility.losses.SparsityPenalty",
)
class SparsityPenaltyConfig(Configuration):
    threshold: float = Param(0.15, ge=0.0, le=1.0)


@register_class(
    name="loss", tags={"classification"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class LossConfig(Configuration):
    """Binds a criterion to the namespace fields it scores."""

    name: str = Param("classification")
    loss: RegistrationKey[th.nn.Module] = Param(CROSS_ENTROPY)
    inputs: List[str] = Param(["class_logits", "y_true"])
    coefficient: float = Param(1.0, ge=0.0)
    enabled: bool = Param(True)


@register_class(
    name="loss",
    tags={"classification", "full"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class FullClassificationLossConfig(LossConfig):
    name: str = Param("full_classification")
    inputs: List[str] = Param(["full_class_logits", "y_true"])


@register_class(
    name="loss",
    tags={"classification", "complement"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class ComplementClassificationLossConfig(LossConfig):
    """What the label looks like from everything the highlight left behind."""

    name: str = Param("complement_classification")
    inputs: List[str] = Param(["complement_class_logits", "y_true"])


@register_class(
    name="loss",
    tags={"classification", "alignment"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class AlignmentClassificationLossConfig(LossConfig):
    """What the label looks like to a module that only ever read full text.

    DAR scores its aligner with this twice: while that module is pretrained on
    the full input, and afterwards on the highlight, where the term is the
    generator's alone.
    """

    name: str = Param("alignment_classification")
    inputs: List[str] = Param(["aligner_class_logits", "y_true"])


@register_class(
    name="loss", tags={"highlight"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class HighlightLossConfig(LossConfig):
    name: str = Param("highlight")
    loss: RegistrationKey[th.nn.Module] = Param(MASKED_CROSS_ENTROPY)
    inputs: List[str] = Param(["highlight_logits", "highlight_true", "mask"])


@register_class(
    name="loss",
    tags={"knowledge", "supervised"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class KnowledgeSupervisionLossConfig(LossConfig):
    """Told which entries to name, with a weight per entry.

    The same supervision as the term below and a different criterion, for one
    reason: two classes under a cross entropy carry a single positive weight,
    and the knowledge axis needs one per entry. The entry that decides a case
    is frequently the rare one, and a shared weight cannot tell it from the
    entry that fires on half the corpus.

    It scores ``knowledge_score``, the difference of the comparer's two logits
    -- the same quantity the gate is taken from, so the term and the gate
    cannot disagree.
    """

    name: str = Param("knowledge")
    loss: RegistrationKey[th.nn.Module] = Param(KNOWLEDGE_BCE)
    inputs: List[str] = Param(["knowledge_score", "knowledge_true", "knowledge_valid"])


@register_class(
    name="loss", tags={"knowledge"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class KnowledgeLossConfig(LossConfig):
    """Which knowledge base entries explain this example, against the gold links.

    The one place in a grounded run where a highlight-side claim meets a gold
    standard. ``knowledge_true`` is ``-1`` on an example the corpus does not
    annotate and ``0`` where it annotates that an entry does not apply, so the
    criterion skips the first and scores the second: an empty knowledge set is
    an answer, not a missing label.
    """

    name: str = Param("knowledge")
    loss: RegistrationKey[th.nn.Module] = Param(MASKED_CROSS_ENTROPY)
    inputs: List[str] = Param(["knowledge_logits", "knowledge_true", "knowledge_valid"])


@register_class(
    name="loss",
    tags={"sparsity", "knowledge"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class KnowledgeSparsityLossConfig(LossConfig):
    """How much of the knowledge base an example is allowed to instantiate.

    The same criterion the token axis uses, bound to the knowledge axis: it
    reads the field names it is given and does not care which axis they are.
    """

    name: str = Param("knowledge_sparsity")
    loss: RegistrationKey[th.nn.Module] = Param(SPARSITY_PENALTY)
    inputs: List[str] = Param(["knowledge_mask", "knowledge_valid"])
    coefficient: float = Param(1.0, ge=0.0)


@register_class(
    name="loss", tags={"sparsity"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class SparsityLossConfig(LossConfig):
    name: str = Param("sparsity")
    loss: RegistrationKey[th.nn.Module] = Param(SPARSITY_PENALTY)
    inputs: List[str] = Param(["highlight_mask", "mask"])


@register_class(
    name="loss", tags={"contiguity"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class ContiguityLossConfig(LossConfig):
    name: str = Param("contiguity")
    loss: RegistrationKey[th.nn.Module] = Param(CONTIGUITY_PENALTY)
    inputs: List[str] = Param(["highlight_mask", "mask"])
    coefficient: float = Param(2.0, ge=0.0)


@register_class(
    name="loss", tags={"discrepancy"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class DiscrepancyLossConfig(LossConfig):
    name: str = Param("discrepancy")
    loss: RegistrationKey[th.nn.Module] = Param(KL_DIV)
    inputs: List[str] = Param(["class_logits", "full_class_logits"])


@register_class(
    name="loss",
    tags={"discrepancy", "remaining"},
    namespace=NAMESPACE,
    component=LOSS_COMPONENT,
)
class RemainingDiscrepancyLossConfig(DiscrepancyLossConfig):
    """MRD's criterion: the complement should stop looking like the whole input.

    Scored between the complement and the full input rather than between the
    highlight and the full input, and **maximized** -- the one term in the
    library a model wants large, which is what the negative coefficient says.
    A coefficient is otherwise non-negative, since a loss is otherwise
    something to minimize.
    """

    name: str = Param("remaining_discrepancy")
    inputs: List[str] = Param(["complement_class_logits", "full_class_logits"])
    coefficient: float = Param(-1.0)


@register_class(
    name="loss", tags={"guide"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class GuideLossConfig(LossConfig):
    name: str = Param("guide")
    loss: RegistrationKey[th.nn.Module] = Param(MASKED_BCE)
    inputs: List[str] = Param(["selection_logits", "guide_target", "mask"])


@register_class(
    name="loss", tags={"jsd"}, namespace=NAMESPACE, component=LOSS_COMPONENT
)
class JSDLossConfig(LossConfig):
    name: str = Param("jsd")
    loss: RegistrationKey[th.nn.Module] = Param(JS_DIV)
    inputs: List[str] = Param(["class_logits", "guider_class_logits"])
