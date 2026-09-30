"""Select-then-predict grounded in a knowledge base.

A corpus may explain its labels with free text rather than with spans. The
ToS-100 corpus of Ruggeri, Lagioia, Lippi and Torroni, 2022, *Detecting and
explaining unfairness in consumer contracts through memory networks*,
Artificial Intelligence and Law 30(1), 59-92,
<https://doi.org/10.1007/s10506-021-09288-2>, records which legal rationales
make a clause unfair, but not which words carry them. The rationales are the
knowledge base ``K = {k_1, ..., k_M}``. Grounding means answering which of
them a given input instantiates, which is a discrete and checkable output.

For every pair ``(x, k_i)`` a highlight pair ``(h_i, h_k_i)`` is extracted: the
words of the input that match the entry, and the words of the entry that match
the input. A comparer scores each pair, and those scores name the subset
``K_x``. The predictor reads the union of the input-side highlights and
classifies from that alone, so the select-then-predict guarantee is unchanged.

:doc:`/models/grounded` describes the model in full.
"""

from __future__ import annotations

import abc
from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.base import SPP, SPPBackbone, probability
from pyhighlights.components.models.spp.data import GroundedSPPOutput
from pyhighlights.components.models.spp.implementations import mlp


class SPPComparer(th.nn.Module, abc.ABC):
    """Scores whether an input highlight instantiates a knowledge highlight."""

    @abc.abstractmethod
    def forward(self, states: th.Tensor, knowledge: th.Tensor) -> th.Tensor:
        """Return link logits shaped ``[B, M, 2]`` from two ``[B, M, D]`` pools."""


class EntailmentComparer(SPPComparer):
    """A directed judgement over ``[u; v; u - v; u * v]``.

    The relation between an input and an entry is instantiation, not
    similarity: the input states a fact, and the entry states the rule the
    fact may instantiate. The scorer is therefore directed, and ``u - v`` is
    what makes it so. ``|u - v|`` would restore the symmetry this exists to
    break.
    """

    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()
        self.comparer = mlp([4 * input_size, *hidden_sizes, 2])

    def forward(self, states: th.Tensor, knowledge: th.Tensor) -> th.Tensor:
        return self.comparer(
            th.cat(
                [
                    states,
                    knowledge,
                    states - knowledge,
                    states * knowledge,
                ],
                dim=-1,
            )
        )


class GroundedSPP(SPP):
    """Select-then-predict over an input and a knowledge base.

    One selector instance serves both sides. The input is read conditioned on
    each knowledge entry and each entry conditioned on the input, and the same
    encoder reads both. Conditioning concatenates the partner's pooled state
    onto the states being scored, which is why the selector is built at twice
    a backbone's output size.

    The knowledge base is a property of the corpus, not of a sample: the same
    entries serve every example of a run. It arrives once through
    :meth:`load_knowledge` and is encoded once per step, so the cost is two
    encoder passes per batch whatever ``M`` is.

    What the predictor reads is the **ungated** union of the input-side
    highlights. :meth:`forward` explains why gating it fails on the majority
    class.
    """

    def __init__(self, comparer: RegistrationKey[SPPComparer], **kwargs):
        super().__init__(**kwargs)
        if len(self.selectors) != 1:
            raise ValueError("GroundedSPP requires exactly one selector")
        self.comparer = Registry.from_key(
            comparer,
            expected_type=SPPComparer,
            input_size=self.selector_backbone.output_size,
        )
        # Not a buffer and not a parameter. The base is text the corpus already
        # ships, so a checkpoint carrying it would store a copy of the corpus
        # and refuse to load into a run configured with a different one. It is
        # moved onto the module's device the first time it is read.
        self._knowledge: InputData | None = None

    def selector_input_size(self, backbone: SPPBackbone) -> int:
        """Twice the width: a selector reads states beside the partner's pool."""
        return 2 * backbone.output_size

    def load_knowledge(self, data: InputData) -> None:
        if data.features.shape[0] < 1:
            raise ValueError("a knowledge base needs at least one entry")
        self._knowledge = data

    def knowledge(self) -> InputData:
        """The base, on this module's device.

        Moved on first read rather than registered as a buffer, and cached
        afterwards, so a run pays one host-to-device copy of a few kilobytes.
        """
        if self._knowledge is None:
            raise ValueError(
                f"{self.name}: no knowledge base was loaded. A grounded model "
                "needs a corpus whose loader defines `knowledge()`"
            )
        if self._knowledge.features.device != self.device:
            self._knowledge = self._knowledge.to(self.device)
        return self._knowledge

    def pool_selection(
        self, states: th.Tensor, selection: th.Tensor, backbone: SPPBackbone
    ) -> th.Tensor:
        """One vector per pair, over the positions that pair selected.

        The pair axis folds into the batch, because pooling is the backbone's
        own operation, a maximum for a recurrent encoder and a masked mean for
        a transformer, and it reads one sequence axis. Reimplementing it here
        would give the comparer a different summary than the predictor sees.
        """
        *leading, width = selection.shape
        pooled = backbone.pool(
            states.reshape(-1, width, states.shape[-1]),
            selection.reshape(-1, width),
        )
        return pooled.reshape(*leading, -1)

    def pairs(
        self, data: InputData, knowledge: InputData
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]:
        """Highlight pairs for every ``(x, k_i)``, and the logits behind them.

        Two encoder passes: the batch once and the base once. Everything after
        them is a head over states already computed, which is what makes
        scoring all ``M`` entries affordable and a retrieval step unnecessary.
        """
        backbone, selector = self.selector_backbone, self.selectors[0]

        states = self.to_words(
            backbone.encode(data.features, self.encoder_mask(data)), data
        )
        entries = self.to_words(
            backbone.encode(knowledge.features, self.encoder_mask(knowledge)),
            knowledge,
        )
        valid = self.selection_valid(data)
        entry_valid = self.selection_valid(knowledge)
        pooled = backbone.pool(states, valid)
        entry_pooled = backbone.pool(entries, entry_valid)

        batch, width, size = states.shape
        entries_count, entry_width, _ = entries.shape
        shape = (batch, entries_count)

        # ponytail: the conditioned states are materialised as [B, M, T, 2D],
        # which is B * M * T * 2D * 4 bytes in float32. Chunk over M, or
        # replace the concatenation with a projected sum, when that exceeds
        # memory.
        pair_logits = selector(
            th.cat(
                [
                    states.unsqueeze(1).expand(*shape, width, size),
                    entry_pooled.view(1, entries_count, 1, size).expand(
                        *shape, width, size
                    ),
                ],
                dim=-1,
            )
        )
        entry_logits = selector(
            th.cat(
                [
                    entries.unsqueeze(0).expand(*shape, entry_width, size),
                    pooled.view(batch, 1, 1, size).expand(*shape, entry_width, size),
                ],
                dim=-1,
            )
        )

        pair_valid = valid.unsqueeze(1).expand(*shape, width)
        entry_pair_valid = entry_valid.unsqueeze(0).expand(*shape, entry_width)
        pair_mask = self.repair_empty(
            pair_logits,
            self.select_activation(pair_logits) * pair_valid,
            pair_valid,
        )
        entry_mask = self.repair_empty(
            entry_logits,
            self.select_activation(entry_logits) * entry_pair_valid,
            entry_pair_valid,
        )

        scores = self.comparer(
            self.pool_selection(
                states.unsqueeze(1).expand(*shape, width, size), pair_mask, backbone
            ),
            self.pool_selection(
                entries.unsqueeze(0).expand(*shape, entry_width, size),
                entry_mask,
                backbone,
            ),
        )
        return pair_logits, pair_mask, entry_mask, scores

    def forward(self, data: InputData) -> GroundedSPPOutput:
        knowledge = self.knowledge()
        pair_logits, pair_mask, entry_mask, scores = self.pairs(data, knowledge)

        # The gate keeps `select_activation` and drops the empty-selection
        # repair that comes with it: an empty knowledge set is the correct
        # answer for an example nothing explains, and repairing it would invent
        # a reason for every one of them.
        gate = self.select_activation(scores)

        # Ungated, and that is the decision the rest of this model rests on.
        # Gating the union collapses on the majority class. An example that
        # gates everything off leaves an empty union, the token-level repair
        # rescues exactly one word, and the predictor learns that one word
        # means negative. The label would be decided by the size of the selection,
        # which is the degeneracy select-then-predict exists to avoid. Ungated,
        # every example carries a full-size highlight under one sparsity
        # target, so a negative one can be read beside a positive one and
        # compared. Each `h_i` is still conditioned on its own entry, so the
        # selection is knowledge-shaped even though the gate does not reach it.
        highlight_mask = pair_mask.amax(dim=1)
        highlight_logits = pair_logits.amax(dim=1)

        return GroundedSPPOutput(
            class_logits=self.predict(
                data=data, highlight_mask=highlight_mask
            ).unsqueeze(1),
            highlight_logits=highlight_logits.unsqueeze(1),
            highlight_mask=highlight_mask.unsqueeze(1),
            knowledge_logits=scores.unsqueeze(1),
            knowledge_score=(scores[..., 1] - scores[..., 0]).unsqueeze(1),
            knowledge_mask=gate.unsqueeze(1),
            knowledge_valid=th.ones_like(gate).unsqueeze(1),
            pair_highlight_mask=pair_mask.unsqueeze(1),
            knowledge_highlight_mask=entry_mask.unsqueeze(1),
        )

    def faithfulness(
        self, input_data: InputData, output_data: GroundedSPPOutput
    ) -> Dict[str, th.Tensor]:
        """Token-level terms, and the two the knowledge axis adds.

        Writing ``K`` for the whole base and ``K_x`` for the entries the model
        named:

        .. code-block:: text

           rationale sufficiency       = p(y_hat | x, K) - p(y_hat | x, K_x)
           rationale comprehensiveness = p(y_hat | x, K) - p(y_hat | x, K \\ K_x)

        Both follow the token-level convention of
        :mod:`pyhighlights.components.faithfulness`: reference minus
        restricted. Lower sufficiency is better, and higher comprehensiveness
        is better. A model whose rationale comprehensiveness is near zero
        predicts the same thing when the entries it named are taken away, so
        its grounding does not carry the prediction.

        The union the predictor reads is ungated, so restricting the base
        restricts which pairs enter that union. The ablation is over which
        entries are present, not over a gate. ``p(y_hat | x, K)`` is the
        model's own output and is read off ``output_data`` rather than
        recomputed.

        An example that named nothing leaves the first union empty, which the
        backbones pool to zeros. That measures a model that grounded the
        example in nothing, and is not a case to repair.
        """
        terms = super().faithfulness(input_data, output_data)
        head = self.reported(output_data)
        gate = head.knowledge_mask.unsqueeze(-1)
        pairs = head.pair_highlight_mask

        predicted = head.class_logits.argmax(dim=-1)
        on_base = probability(head.class_logits, predicted)
        on_named = probability(
            self.predict(data=input_data, highlight_mask=(pairs * gate).amax(dim=1)),
            predicted,
        )
        on_rest = probability(
            self.predict(
                data=input_data, highlight_mask=(pairs * (1 - gate)).amax(dim=1)
            ),
            predicted,
        )
        return {
            **terms,
            "rationale_sufficiency": on_base - on_named,
            "rationale_comprehensiveness": on_base - on_rest,
        }
