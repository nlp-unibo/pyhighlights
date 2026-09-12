from __future__ import annotations

import math
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry


class Loss(th.nn.Module):
    """Binds a criterion to the fields feeding it.

    The criterion is any ``th.nn.Module`` taking tensors, so it stays reusable
    across bindings: the same criterion can score different pairs of fields by
    registering it twice with different ``inputs``.
    """

    def __init__(
        self,
        name: str,
        loss: RegistrationKey[th.nn.Module],
        inputs: Sequence[str],
        coefficient: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__()
        if not inputs:
            raise ValueError(f"Loss {name} requires at least one input field")
        self.name = name
        self.loss = Registry.from_key(loss, expected_type=th.nn.Module)
        self.inputs = list(inputs)
        self.coefficient = coefficient
        self.enabled = enabled

    def forward(self, values: Mapping[str, th.Tensor]) -> th.Tensor:
        missing = [name for name in self.inputs if name not in values]
        if missing:
            raise KeyError(f"Loss {self.name} misses input fields {missing}")
        return self.loss(*(values[name] for name in self.inputs))


def build_losses(keys: List[RegistrationKey[Loss]]) -> List[Loss]:
    return Registry.from_keys(keys, expected_type=Loss)


def compute_losses(
    losses: Iterable[Loss],
    values: Mapping[str, th.Tensor],
    scales: Mapping[str, float] | None = None,
) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
    """Sum enabled losses over ``values``, reporting each term unscaled.

    ``scales`` applies an extra per-name factor on top of the loss coefficient,
    which is how annealed terms are weighted without mutating the loss.
    """
    total = None
    computed: Dict[str, th.Tensor] = {}

    for loss in losses:
        if not loss.enabled:
            continue
        value = loss(values)
        computed[loss.name] = value
        scale = loss.coefficient * (scales.get(loss.name, 1.0) if scales else 1.0)
        total = value * scale if total is None else total + value * scale

    if total is None:
        reference = next(
            (value for value in values.values() if isinstance(value, th.Tensor)), None
        )
        total = th.zeros(()) if reference is None else reference.new_zeros(())

    return total, computed


class CrossEntropy(th.nn.Module):
    """Cross entropy over class logits, with per-class weights.

    ``th.nn.CrossEntropyLoss`` takes its weights as a tensor, and no
    configuration should carry a tensor. A list of per-class weights is not a
    tensor's worth of data -- one number per class -- so that is what gets
    declared, and the tensor is built here.

    The weights are a buffer rather than an attribute, so they follow the model
    onto whatever device it moves to. A weight tensor left on the CPU is a
    crash at the first batch of a GPU run.

    Weights are declared rather than computed. A corpus whose split is fixed
    has fixed class frequencies, so the numbers a run needs are known before it
    starts -- and writing them down puts them in the manifest, where a
    balanced weighting computed inside the run would leave nothing.
    """

    def __init__(self, weight: Sequence[float] | None = None, ignore_index: int = -100):
        super().__init__()
        # Not persistent: the weights come from the configuration, not from
        # training, so a checkpoint that carried them would refuse to load
        # into a run configured without them.
        self.register_buffer(
            "weight",
            None
            if weight is None
            else th.tensor(list(weight), dtype=th.get_default_dtype()),
            persistent=False,
        )
        self.ignore_index = ignore_index

    def forward(self, logits: th.Tensor, targets: th.Tensor) -> th.Tensor:
        return th.nn.functional.cross_entropy(
            logits,
            targets.long(),
            weight=self.weight,
            ignore_index=self.ignore_index,
        )


class MaskedCrossEntropy(th.nn.Module):
    """Cross entropy over valid, labelled positions of any axis.

    Positions are whatever the binding names: tokens of a sequence, or entries
    of a knowledge base. Everything before the class axis is flattened, so the
    shape of the thing being scored is the binding's business.
    """

    def __init__(self, ignore_index: int = -1, weight: Sequence[float] | None = None):
        """``weight`` is one factor per class, for an axis that is imbalanced.

        The knowledge axis is: a clause instantiates one or two of up to 28
        rationales, so the negative class outnumbers the positive one by more
        than an order of magnitude before the fair clauses -- which instantiate
        nothing at all -- are counted.

        Not persistent, like :class:`CrossEntropy`'s: the weights come from the
        configuration rather than from training.
        """
        super().__init__()
        self.register_buffer(
            "weight",
            None
            if weight is None
            else th.tensor(list(weight), dtype=th.get_default_dtype()),
            persistent=False,
        )
        self.ignore_index = ignore_index

    def forward(
        self, logits: th.Tensor, targets: th.Tensor, mask: th.Tensor
    ) -> th.Tensor:
        logits = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1)
        valid = (targets != self.ignore_index) & mask.reshape(-1).bool()
        if not valid.any():
            return logits.sum() * 0
        return th.nn.functional.cross_entropy(
            logits[valid], targets[valid].long(), weight=self.weight
        )


class MaskedBinaryCrossEntropy(th.nn.Module):
    """Binary cross entropy with logits over valid, labelled positions.

    One independent decision per position, which is what the knowledge axis
    is: a knowledge base entry either explains an example or it does not, and
    several may.
    """

    def __init__(
        self,
        ignore_index: int = -1,
        pos_weight: Sequence[float] | None = None,
    ):
        """``pos_weight`` is what a shared cross entropy cannot express.

        One factor per position of the axis being scored, broadcast over
        everything in front of it, multiplying the cost of missing a positive
        there. Two classes under :class:`MaskedCrossEntropy` carry one global
        weight; this carries a different weight per knowledge base entry, and
        that is the difference that matters -- the decisive entry is often the
        rare one, and a single positive weight cannot tell it apart from the
        entry that fires on half the corpus.

        Read it off the corpus rather than typing it:
        :class:`~pyhighlights.components.preprocessors.KnowledgeWeights`.

        ``ignore_index`` marks a position carrying no annotation. It was not
        skipped before this, which was safe only because nothing bound this
        criterion to a field that uses the marker -- a ``-1`` reaching a
        binary target is not a label, it is a number the loss would happily
        descend on.
        """
        super().__init__()
        # Not persistent, like every other configured weight here.
        self.register_buffer(
            "pos_weight",
            None
            if pos_weight is None
            else th.tensor(list(pos_weight), dtype=th.get_default_dtype()),
            persistent=False,
        )
        self.ignore_index = ignore_index

    def forward(
        self, logits: th.Tensor, targets: th.Tensor, mask: th.Tensor
    ) -> th.Tensor:
        valid = mask.bool() & (targets != self.ignore_index)
        if not valid.any():
            return logits.sum() * 0
        weight = self.pos_weight
        if weight is not None:
            if weight.shape[0] != targets.shape[-1]:
                raise ValueError(
                    f"pos_weight has {weight.shape[0]} entries but the axis "
                    f"being scored has {targets.shape[-1]}"
                )
            weight = weight.expand_as(targets)[valid]
        return th.nn.functional.binary_cross_entropy_with_logits(
            logits[valid], targets[valid].to(logits.dtype), pos_weight=weight
        )


class SparsityPenalty(th.nn.Module):
    """Distance between the selection rate and a target rate."""

    def __init__(self, threshold: float = 0.15):
        super().__init__()
        self.threshold = threshold

    def forward(self, selection: th.Tensor, mask: th.Tensor) -> th.Tensor:
        rate = selection.sum() / mask.sum().clamp_min(1)
        return th.abs(rate - self.threshold)


class ContiguityPenalty(th.nn.Module):
    """Mean absolute transition between neighbouring valid selections."""

    def forward(self, selection: th.Tensor, mask: th.Tensor) -> th.Tensor:
        if selection.shape[-1] < 2:
            return selection.sum() * 0

        valid = mask[..., 1:].bool() & mask[..., :-1].bool()
        differences = th.abs(selection[..., 1:] - selection[..., :-1])
        if not valid.any():
            return differences.sum() * 0
        return differences[valid].mean()


class KLDiv(th.nn.Module):
    def forward(self, p: th.Tensor, q: th.Tensor) -> th.Tensor:
        return th.nn.functional.kl_div(
            th.nn.functional.log_softmax(p, dim=-1),
            th.nn.functional.softmax(q, dim=-1),
            reduction="batchmean",
        )


class JSDiv(th.nn.Module):
    def forward(self, p: th.Tensor, q: th.Tensor) -> th.Tensor:
        log_p = th.nn.functional.log_softmax(p, dim=-1)
        log_q = th.nn.functional.log_softmax(q, dim=-1)
        log_mean = th.logaddexp(log_p, log_q) - math.log(2)
        return (
            th.nn.functional.kl_div(
                log_mean, log_p, reduction="batchmean", log_target=True
            )
            + th.nn.functional.kl_div(
                log_mean, log_q, reduction="batchmean", log_target=True
            )
        ) / 2
