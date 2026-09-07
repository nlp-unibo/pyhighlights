from __future__ import annotations

from typing import List, Mapping, Sequence

import torch as th
from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric


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
