from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Generator, TypeVar

import torch as th

D = TypeVar("D", bound="ModelData")


@dataclass
class ModelData:
    def as_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}

    def as_numpy(self):
        return {
            field.name: value.detach().cpu().numpy()
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), th.Tensor)
        }

    def unbind(self: D, dim: int = 0) -> Generator[D, None, None]:
        tensor_fields = {
            field.name: th.unbind(value, dim=dim)
            for field in fields(self)
            if isinstance((value := getattr(self, field.name)), th.Tensor)
        }
        sizes = {len(values) for values in tensor_fields.values()}
        if not sizes:
            raise RuntimeError("No tensors found to unbind")
        if len(sizes) != 1:
            raise RuntimeError("Cannot unbind tensors of different sizes")

        for index in range(sizes.pop()):
            yield type(self)(
                **{
                    field.name: tensor_fields[field.name][index]
                    if field.name in tensor_fields
                    else getattr(self, field.name)
                    for field in fields(self)
                }
            )


@dataclass
class InputData(ModelData):
    features: th.Tensor
    mask: th.Tensor
    sample_ids: th.Tensor
    y_true: th.Tensor
    highlight_true: th.Tensor
    #: Which source word each position came from, ``-1`` where none did. No
    #: model reads it: it is what turns a selection over subtokens back into a
    #: selection over words, which is the only form a person can read. Optional
    #: because a batch assembled by hand has no words behind it.
    word_ids: th.Tensor | None = None


@dataclass
class OutputData(ModelData):
    class_logits: th.Tensor
