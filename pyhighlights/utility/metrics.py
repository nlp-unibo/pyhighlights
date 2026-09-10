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
    def compute(self) -> th.Tensor:
        return (2 * self.tp) / (2 * self.tp + self.fp + self.fn)


class BinaryHighlightIoU(HighlightMetric):
    def compute(self) -> th.Tensor:
        return self.tp / (self.tp + self.fp + self.fn)


class SelectionMetric(Metric):
    """Per-sample selection statistic averaged over the labelled positions."""

    is_differentiable = False
    higher_is_better = False
    full_state_update = False

    def __init__(self, ignore_index: int = -1, **kwargs):
        super().__init__(**kwargs)

        self.ignore_index = ignore_index
        self.add_state(
            name="value", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
        )
        self.add_state(
            name="samples", default=th.tensor(0, dtype=th.float), dist_reduce_fx="sum"
        )

    def reduce(self, selection: th.Tensor) -> th.Tensor:
        raise NotImplementedError

    def update(self, preds: th.Tensor, target: th.Tensor) -> None:
        for sample_preds, sample_target in zip(preds, target):
            selection = sample_preds[sample_target != self.ignore_index]
            if selection.numel():
                self.value += self.reduce(selection).detach()
                self.samples += 1

    def compute(self) -> th.Tensor:
        return self.value / self.samples if self.samples > 0 else self.value


class SelectionRate(SelectionMetric):
    def reduce(self, selection: th.Tensor) -> th.Tensor:
        return selection.mean()


class SelectionSize(SelectionMetric):
    def reduce(self, selection: th.Tensor) -> th.Tensor:
        return selection.sum()
