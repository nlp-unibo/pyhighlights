"""What ties a component to the fields of a step that feed it.

A loss and a metric are the same shape of thing: a scoring object, a name to
report it under, and the names of the fields it reads. The step namespace is a
flat dictionary of tensors, built from the batch and the model output, so the
binding is what decides which of those tensors reach the object and in which
order. That is why the same cross entropy scores a class axis or a knowledge
axis, and why the same F1 scores classes or tokens, without either of them
knowing which model produced the fields.

Binding here rather than in each of them, because two copies of one contract
drift: the checks, the error messages and the order of arguments have to agree
for a study to read a failure in either.
"""

from __future__ import annotations

from typing import Mapping, Sequence, Tuple

import torch as th

__all__ = ["Binding"]


class Binding(th.nn.Module):
    """A named component reading named fields of the step namespace."""

    #: What a subclass calls itself in an error message, so a study reads
    #: ``Loss classification`` or ``Metric highlight_f1`` rather than a word
    #: that names neither.
    kind: str = "Binding"

    def __init__(self, name: str, inputs: Sequence[str]):
        super().__init__()
        if not inputs:
            raise ValueError(f"{self.kind} {name} requires at least one input field")
        self.name = name
        self.inputs = list(inputs)

    def arguments(self, values: Mapping[str, th.Tensor]) -> Tuple[th.Tensor, ...]:
        """The bound fields, in the order the object reads them.

        Every field is checked before any of them is read, so a binding that
        names two missing fields says both rather than the first one.
        """
        missing = [name for name in self.inputs if name not in values]
        if missing:
            raise KeyError(f"{self.kind} {self.name} misses input fields {missing}")
        return tuple(values[name] for name in self.inputs)
