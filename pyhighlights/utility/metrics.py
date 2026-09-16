from __future__ import annotations

from typing import List, Mapping, Sequence

import torch as th
from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric
from torchmetrics.classification import MulticlassF1Score


class BoundMetric(th.nn.Module):
    """Binds a ``torchmetrics`` metric to the fields feeding it.

    Mirrors ``Loss``: the metric stays a plain ``Metric``, and the binding says
    which fields of the step namespace it scores.
    """

    def __init__(
        self,
        name: str,
        metric: RegistrationKey[Metric],
        inputs: Sequence[str] = ("class_logits", "y_true"),
    ):
        super().__init__()
        if not inputs:
            raise ValueError(f"Metric {name} requires at least one input field")
        self.name = name
        self.metric = Registry.from_key(metric, expected_type=Metric)
        self.inputs = list(inputs)

    def update(self, values: Mapping[str, th.Tensor]) -> None:
        missing = [name for name in self.inputs if name not in values]
        if missing:
            raise KeyError(f"Metric {self.name} misses input fields {missing}")
        self.metric.update(*(values[name] for name in self.inputs))

    def compute(self) -> th.Tensor:
        return self.metric.compute()

    def reset(self) -> None:
        self.metric.reset()


def build_metrics(
    keys: List[RegistrationKey[BoundMetric]] | None,
) -> th.nn.ModuleList:
    return th.nn.ModuleList(Registry.from_keys(keys or [], expected_type=BoundMetric))


class ClassF1Score(MulticlassF1Score):
    """F1 of one class rather than an average over all of them.

    Macro F1 over two classes is half the majority class, and on a corpus
    where one class is 99.5% of the rows that half carries the score: a model
    that answers "negative" to everything reports 0.50 while finding nothing.
    Averaging is the wrong summary there -- what the run is about is the rare
    class, so this reports it alone.

    ``pos_label`` names that class. The predictor emits one logit per class,
    which is why this is a multiclass metric restricted to one class rather
    than ``task="binary"``: binary wants one score per sample.
    """

    def __init__(self, pos_label: int = 1, num_classes: int = 2, **kwargs):
        if not 0 <= pos_label < num_classes:
            raise ValueError(f"pos_label {pos_label} is not a class of {num_classes}")
        # `average` is not forwarded: passing one raises rather than being
        # ignored, since a caller who asks for an average is asking for the
        # metric this one exists to replace.
        super().__init__(num_classes=num_classes, average="none", **kwargs)
        self.pos_label = pos_label

    def compute(self) -> th.Tensor:
        return super().compute()[self.pos_label]


class HighlightMetric(Metric):
    """Token-level confusion counts over labelled highlight positions."""

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    def __init__(self, pos_label: int = 1, ignore_index: int = -1, **kwargs):
        super().__init__(**kwargs)

        self.pos_label = pos_label
        self.ignore_index = ignore_index
        for state in ("tp", "fp", "fn"):
            self.add_state(
                name=state, default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
            )

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        valid = target != self.ignore_index
        predicted = (preds == self.pos_label) & valid
        expected = (target == self.pos_label) & valid

        self.tp += (predicted & expected).sum()
        self.fp += (predicted & ~expected & valid).sum()
        self.fn += (~predicted & expected).sum()


class BinaryHighlightF1Score(HighlightMetric):
    """``nan`` where there is nothing to score, deliberately.

    The denominator is zero when no annotated position was marked, either by
    the corpus or by the model. Two ways to reach it: the metric was never
    updated, or every position it saw was a true negative -- a split with no
    annotations, or one whose rows annotate that nothing is a highlight and a
    model that selected nothing on them.

    Both are the same statement -- no highlight was asked for and none was
    offered -- and ``nan`` is what says it. Returning 0.0, which is what
    torchmetrics' own ``zero_division`` default does, would be a *score*, and a
    score of zero says the model got everything wrong. Returning 1.0 would say
    it got everything right for selecting nothing. Neither happened.

    An earlier comment here claimed an unannotated row still counts a false
    positive, so that only an unused metric could divide by zero. It does not:
    :meth:`HighlightMetric.update` masks predictions by ``valid``, so a row the
    corpus does not annotate contributes to no counter at all.
    """

    def compute(self) -> th.Tensor:
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)


class BinaryHighlightIoU(HighlightMetric):
    """``nan`` where there is nothing to score; see :class:`BinaryHighlightF1Score`."""

    def compute(self) -> th.Tensor:
        return self.tp / (self.tp + self.fp + self.fn)


class BinaryHighlightPrecision(HighlightMetric):
    """Of the positions the selection marked, the share the corpus annotates.

    Reported beside recall because a sparsity target moves the two in opposite
    directions and their F1 hides it: a selector squeezed below the annotation's
    own length buys precision with recall, and a column of F1 alone reads as a
    model that got slightly worse rather than one that changed what it does.

    ``nan`` where there is nothing to score; see :class:`BinaryHighlightF1Score`.
    Here the denominator is zero when the model marked no annotated position,
    which is a different statement from marking the wrong ones.
    """

    def compute(self) -> th.Tensor:
        return self.tp / (self.tp + self.fp)


class BinaryHighlightRecall(HighlightMetric):
    """Of the positions the corpus annotates, the share the selection marked.

    ``nan`` where there is nothing to score; see :class:`BinaryHighlightF1Score`.
    The denominator is zero on a split that annotates nothing.
    """

    def compute(self) -> th.Tensor:
        return self.tp / (self.tp + self.fn)


class SelectionMetric(Metric):
    """Per-sample selection statistic over the tokens a document actually has.

    ``target`` is the **padding mask** -- 1 for a real token, 0 for padding --
    and not an annotation. A selection statistic is about the document, so it
    is defined whether or not the corpus annotated anything, which is why the
    registered binding names ``mask`` rather than ``highlight_true``.

    The distinction is the whole metric. Counting padding makes a rate depend
    on the widest row in the batch rather than on the document: a 6-token
    selection out of a 34-token clause is 18%, and reads as 6% once 67 columns
    of padding join the denominator. Sizes survive that -- padding adds zero to
    a sum -- and rates do not.
    """

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.add_state(
            name="value", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
        )
        self.add_state(
            name="samples", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
        )

    def reduce(self, selected: th.Tensor, length: th.Tensor) -> th.Tensor:
        """One statistic per document, from what it kept and how long it is.

        ``selected`` is ``[B, T]`` with every padded position already zeroed,
        and ``length`` is ``[B]``; both cover only the documents that have a
        token. A subclass sums, divides, or reads the positions themselves --
        a count of spans is not recoverable from a total, which is why this
        takes the row rather than its sum.
        """
        raise NotImplementedError

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        # Over the batch rather than a row at a time. The loop this replaces
        # cost 2.36 ms per batch of 64 against a 54 ms training step, all of it
        # Python: the arithmetic is two masked sums.
        valid = target > 0
        length = valid.sum(dim=-1)
        # A row of pure padding has no selection rate. Excluded rather than
        # counted as zero, which is what the loop did by skipping it.
        counted = length > 0
        if not counted.any():
            return
        # `where` rather than a multiply: a padded position is dropped whatever
        # it holds, and `0 * nan` is `nan`. The loop this replaces never looked
        # at those positions, so neither does this.
        selected = th.where(valid, preds, th.zeros_like(preds))
        self.value += self.reduce(selected[counted], length[counted]).sum().detach()
        # On the device, like the states themselves: `int()` here would force a
        # host synchronisation on every batch, which the loop never did.
        self.samples += counted.sum()

    def compute(self) -> th.Tensor:
        return self.value / self.samples if self.samples > 0 else self.value


class SelectionRate(SelectionMetric):
    """Share of a document the selection kept, so bounded by zero and one."""

    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    def reduce(self, selected: th.Tensor, length: th.Tensor) -> th.Tensor:
        return selected.sum(dim=-1) / length


class SelectionSize(SelectionMetric):
    """How many tokens the selection kept, which nothing bounds above."""

    def reduce(self, selected: th.Tensor, length: th.Tensor) -> th.Tensor:
        return selected.sum(dim=-1)


class SelectionSpans(SelectionMetric):
    """How many contiguous runs the selection falls into.

    Contiguity is a penalty in :mod:`pyhighlights.utility.losses` and was never
    a reported number, so a run said how much it kept and never whether the
    kept words sit together. Six words in one span and six scattered over a
    clause are the same selection size and not the same highlight: the first
    can be read as a phrase, the second is what a model keying on punctuation
    produces.

    A run of ones is counted at its first position, so a document that selects
    nothing scores zero and one that selects everything scores one.

    **The mask has to be hard.** Every positive entry opens or continues a run,
    so a probability of 0.01 counts as kept. :class:`SelectionRate` and
    :class:`SelectionSize` sum their input and stay meaningful on a soft mask;
    this one does not, and reports the number of runs of non-zero entries
    rather than anything about the highlight a threshold would produce.
    """

    def reduce(self, selected: th.Tensor, length: th.Tensor) -> th.Tensor:
        kept = selected > 0
        # A position starts a run when it is kept and its predecessor is not.
        # The pad supplies the missing predecessor of column zero as "not
        # kept", so a selection beginning at the first token is one span.
        previous = th.nn.functional.pad(kept, (1, 0))[..., :-1]
        return (kept & ~previous).sum(dim=-1).to(selected.dtype)


class KnowledgeSetMetric(Metric):
    """Per-example agreement between the entries named and the entries annotated.

    Both scores here are about the **set**, not about the individual links.
    Per-link F1 rewards naming one correct entry out of several and stopping;
    what a reader wants to know is whether the model got the explanation
    right, and a partial set is a partial explanation.

    An example the corpus does not annotate scores nothing: ``ignore_index``
    marks it, and it is skipped rather than counted as an empty set. That
    distinction is the whole point of the knowledge axis carrying ``-1`` and
    ``0`` as different values.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = False
    plot_lower_bound = 0.0
    plot_upper_bound = 1.0

    def __init__(self, threshold: float = 0.5, ignore_index: int = -1, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.ignore_index = ignore_index
        for state in ("hits", "examples"):
            self.add_state(
                name=state, default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
            )

    def counts(self, preds: th.Tensor, target: th.Tensor):
        """Predicted and annotated sets, over the annotated examples only."""
        annotated = (target != self.ignore_index).all(dim=-1)
        return (
            (preds[annotated] > self.threshold),
            (target[annotated] > 0),
        )

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        raise NotImplementedError

    def compute(self) -> th.Tensor:
        return self.hits / self.examples if self.examples > 0 else self.hits


class ExactSetMatch(KnowledgeSetMetric):
    """Share of examples whose named set is exactly the annotated one.

    Strict, and the number a domain expert cares about: naming two of the
    three rationales that make a clause unfair explains it two thirds of the
    way, which is not what an explanation is for.
    """

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        predicted, expected = self.counts(preds, target)
        self.hits += (predicted == expected).all(dim=-1).sum()
        self.examples += predicted.shape[0]


class EmptySetAccuracy(KnowledgeSetMetric):
    """Share of examples annotated with nothing that were named nothing.

    Reported on its own rather than folded into an average, because the
    examples that instantiate nothing are the overwhelming majority -- 97.7% of
    ToS-100 -- and would otherwise carry every number they entered. An example
    grounded in an entry it has no business in has invented a reason, which is
    a scoring error and not an ambiguity.
    """

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        predicted, expected = self.counts(preds, target)
        empty = ~expected.any(dim=-1)
        self.hits += (~predicted[empty].any(dim=-1)).sum()
        # On the device, like the state it adds to: `int()` here forces a host
        # synchronisation on every batch.
        self.examples += empty.sum()
