from __future__ import annotations

from dataclasses import dataclass

import torch as th

from pyhighlights.components.models.data import OutputData


@dataclass
class SPPOutput(OutputData):
    highlight_logits: th.Tensor
    highlight_mask: th.Tensor
