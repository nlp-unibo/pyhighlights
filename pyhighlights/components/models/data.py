from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Generator, TypeVar

import torch as th

D = TypeVar("D", bound="ModelData")


@dataclass
class ModelData:
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


@dataclass
class OutputData(ModelData):
    class_logits: th.Tensor

    @property
    def y_pred(self) -> th.Tensor:
        """Compatibility alias for pre-0.2 code."""
        return self.class_logits


@dataclass
class SPPOutput(OutputData):
    highlight_logits: th.Tensor
    highlight_mask: th.Tensor

    @property
    def highlight_pred(self) -> th.Tensor:
        """Compatibility alias for pre-0.2 code."""
        return self.highlight_mask
