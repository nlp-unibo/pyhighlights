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

    def to(self: D, device: th.device | str) -> D:
        """The same record with every tensor on ``device``, non-tensors as they are."""
        return type(self)(
            **{
                name: value.to(device) if isinstance(value, th.Tensor) else value
                for name, value in self.as_dict().items()
            }
        )

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
    """A batch on two axes, and which one a field lives on matters.

    The **subtoken axis** ``[B, T]`` is what the encoder reads: ``features``,
    ``attention_mask`` and ``word_ids``. A backbone tokenizes however it likes,
    so ``T`` is its business.

    The **word axis** ``[B, W]`` is what a person reads and what a selection is
    made over: ``mask`` and ``highlight_true``. A word is the unit the corpus
    annotates, the unit a sparsity target is a fraction of, and the unit an
    export shows -- and it is the same unit whichever backbone read the text,
    which is what makes two backbones comparable in one table.

    ``word_ids`` is the map between them. For a vocabulary tokenizer the two
    axes coincide and it is the identity.
    """

    features: th.Tensor
    #: Word axis: which word slots hold a word. Losses and metrics bind here.
    mask: th.Tensor
    sample_ids: th.Tensor
    y_true: th.Tensor
    #: Word axis: the annotation, ``-1`` where a split carries none.
    highlight_true: th.Tensor
    #: Subtoken axis: which word each position came from, ``-1`` for a special
    #: token or padding. The map between the axes, and what turns a selection
    #: back into something readable.
    word_ids: th.Tensor | None = None
    #: Subtoken axis: what the encoder attends over -- every real subtoken
    #: *and* every special token, since a backbone was pretrained with those
    #: and reads worse without them. Distinct from ``mask``, which says what
    #: may be selected: ``[CLS]`` is not a word and is never a choice, but it
    #: is always seen. Defaults to every non-padding position when a batch is
    #: assembled by hand.
    attention_mask: th.Tensor | None = None

    def attention(self) -> th.Tensor:
        """What the encoder attends over, whether or not the batch said."""
        if self.attention_mask is not None:
            return self.attention_mask
        if self.word_ids is not None:
            return (self.word_ids >= 0).to(self.features.dtype)
        return th.ones_like(self.features, dtype=th.get_default_dtype())


@dataclass
class OutputData(ModelData):
    class_logits: th.Tensor
