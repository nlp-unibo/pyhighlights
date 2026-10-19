"""EVAL-X: an evaluator that reads selections without learning a selector's code.

A predictor trained on one selector's selections can learn that selector's
code. On the toy corpus, every model classifies from selections that rarely
contain the highlight. EVAL-X is an evaluator trained on random selections of
the input with the document's own label. It estimates the label given any
subset of the input, and no selector's code reaches its training data.

Jethani, Sudarshan, Aphinyanaphongs and Ranganath, 2021, *Have We Learned to
Explain? How Interpretability Methods Can Learn to Encode Predictions in their
Interpretations*, AISTATS 2021, PMLR 130:1459-1467.

The evaluator is an :class:`~pyhighlights.components.models.spp.base.SPP`
whose selector is :class:`RandomSelector`. It reads a selection through
:meth:`~pyhighlights.components.models.spp.base.SPP.predict`, with the same
backbones and the same masking as the predictors it evaluates.
:func:`~pyhighlights.components.codes.evaluator_readings` scores a model's
stored selections with a trained evaluator.

The keep rate is drawn per example rather than fixed, so that small
selections, where codes live, are covered. The selector is random at
validation and test as well, so a run's metrics describe random selections
of each split.
"""

from __future__ import annotations

from typing import Tuple

import torch as th

from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp.base import SPP, SPPBackbone, SPPSelector

MODES = ("tokens", "spans")


class RandomSelector(SPPSelector):
    """Keeps a random share of each example's valid positions.

    The share is drawn per example, uniformly from ``[min_keep, max_keep]``.
    In ``"tokens"`` mode, each valid position is kept independently with that
    probability. In ``"spans"`` mode, one contiguous span of that share of the
    valid positions is kept, at a uniformly drawn start. Every example with a
    valid position keeps at least one, chosen uniformly, so the model never
    has to repair an empty selection.

    It has no parameters and reads no states. The valid positions have to
    form a prefix of each row, as they do on the word axis of a padded batch.
    """

    def __init__(
        self,
        input_size: int | None = None,
        mode: str = "tokens",
        min_keep: float = 0.0,
        max_keep: float = 1.0,
    ):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        if not 0.0 <= min_keep <= max_keep <= 1.0:
            raise ValueError("keep rates must satisfy 0 <= min_keep <= max_keep <= 1")
        self.mode = mode
        self.min_keep = min_keep
        self.max_keep = max_keep

    def forward(self, states: th.Tensor) -> th.Tensor:
        raise NotImplementedError("RandomSelector reads no states; call sample")

    def sample(self, valid: th.Tensor) -> th.Tensor:
        """A ``0``/``1`` mask shaped like ``valid``, kept only where valid."""
        valid = valid.bool()
        rows, width = valid.shape
        rate = self.min_keep + (self.max_keep - self.min_keep) * th.rand(
            rows, 1, device=valid.device
        )
        count = valid.sum(dim=-1, keepdim=True)
        positions = th.arange(width, device=valid.device).expand(rows, width)
        if self.mode == "tokens":
            keep = th.rand(rows, width, device=valid.device) < rate
        else:
            length = (rate * count).ceil().clamp(min=1)
            room = count - length + 1
            start = (th.rand(rows, 1, device=valid.device) * room).floor()
            keep = (positions >= start) & (positions < start + length)
        keep = keep & valid
        empty = ~keep.any(dim=-1, keepdim=True) & (count > 0)
        chosen = (th.rand(rows, 1, device=valid.device) * count).floor()
        keep = keep | (empty & (positions == chosen))
        return keep.float()


class EvalX(SPP):
    """An evaluator trained on random selections with the document's label.

    One :class:`RandomSelector` replaces the learned selector. Selecting skips
    the encoder pass that a learned selector needs, so the backbone runs once
    per batch, for the predictor.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if len(self.selectors) != 1 or not isinstance(
            self.selectors[0], RandomSelector
        ):
            raise ValueError("EvalX requires exactly one RandomSelector")

    def select(
        self,
        data: InputData,
        selector: SPPSelector,
        backbone: SPPBackbone,
    ) -> Tuple[th.Tensor, th.Tensor]:
        highlight_mask = selector.sample(self.selection_valid(data))
        highlight_logits = th.stack([1.0 - highlight_mask, highlight_mask], dim=-1)
        return highlight_logits, highlight_mask
