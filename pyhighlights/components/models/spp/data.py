from __future__ import annotations

from dataclasses import dataclass

import torch as th

from pyhighlights.components.models.data import OutputData


@dataclass
class SPPOutput(OutputData):
    highlight_logits: th.Tensor
    highlight_mask: th.Tensor


@dataclass
class GroundedSPPOutput(SPPOutput):
    """An SPP output plus the knowledge axis that produced it.

    Every field carries the head axis the aggregator collapses, so a grounded
    output unbinds exactly as an ungrounded one does.

    The per-pair masks are kept rather than folded into the aggregate on
    purpose: a reader wants what the model proposed against *each* knowledge
    entry, and for an example nothing explains that is the only way to see why
    none of them fired.
    """

    #: ``[B, H, M, 2]``: how strongly each entry is instantiated.
    knowledge_logits: th.Tensor
    #: ``[B, H, M]``: one score per entry, positive where it is instantiated.
    #:
    #: The difference of the two logits above, which is what a binary criterion
    #: reads and what a threshold is swept over. Kept as a field rather than
    #: recomputed by every reader: a score and the gate taken from it must not
    #: be able to disagree.
    knowledge_score: th.Tensor
    #: ``[B, H, M]``: the entries the model named, ``K_x``.
    knowledge_mask: th.Tensor
    #: ``[B, H, M]``: which columns hold an entry, for a criterion to bind to.
    knowledge_valid: th.Tensor
    #: ``[B, H, M, T]``: the input-side highlight of each pair.
    pair_highlight_mask: th.Tensor
    #: ``[B, H, M, T_k]``: the knowledge-side highlight of each pair.
    knowledge_highlight_mask: th.Tensor
