from __future__ import annotations

import abc
import math
from typing import Dict, List, Literal, Tuple, Union

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, Model, Split
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility.losses import Loss, compute_losses


class SPPBackbone(th.nn.Module, abc.ABC):
    """Backend-specific token encoder and pooler."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """Token and pooled state width."""

    @abc.abstractmethod
    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return token states shaped [B, T, D]."""

    @abc.abstractmethod
    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """Return sequence states shaped [B, D]."""

    def load_embeddings(self, matrix: th.Tensor) -> None:
        """Adopt a pretrained token embedding table.

        Optional: a backbone whose tokens are already embedded by something
        else -- a pretrained Transformer, say -- has nothing to load, and says
        so rather than silently ignoring the tensor it was handed.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not take a token embedding matrix"
        )


class SPPSelector(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, states: th.Tensor) -> th.Tensor:
        """Return selection logits shaped [B, T, 2]."""


class SPPPredictor(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, states: th.Tensor) -> th.Tensor:
        """Return class logits shaped [B, C]."""


class SPPAggregator(th.nn.Module, abc.ABC):
    """Collapses the head axis of an ``SPPOutput`` into a single head."""

    @abc.abstractmethod
    def forward(self, output_data: SPPOutput) -> SPPOutput: ...


class SPPFirstAggregator(SPPAggregator):
    def forward(self, output_data: SPPOutput) -> SPPOutput:
        return next(output_data.unbind(dim=1))


class SPP(Model[SPPOutput]):
    def __init__(
        self,
        selector_backbones: Union[
            RegistrationKey[SPPBackbone], List[RegistrationKey[SPPBackbone]]
        ],
        selectors: Union[
            RegistrationKey[SPPSelector], List[RegistrationKey[SPPSelector]]
        ],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None = None,
        aggregator: RegistrationKey[SPPAggregator] | None = None,
        temperature: float = 1.0,
        select_over: Literal["word", "subtoken"] = "word",
        supervise_highlights: bool = False,
        highlight_loss: RegistrationKey[Loss] | None = None,
        highlight_coefficient: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Supervision is a setting, not a variant: the same model key runs
        # unsupervised or guided by the annotation, and the two are different
        # experiments rather than two points on one scale -- the guided one is
        # the ceiling the unsupervised one is measured against.
        self.supervised = None
        if supervise_highlights:
            if highlight_loss is None:
                raise ValueError(
                    "supervise_highlights needs a highlight loss to supervise with"
                )
            self.losses.append(
                Registry.from_key(
                    highlight_loss,
                    expected_type=Loss,
                    coefficient=highlight_coefficient,
                )
            )
            self.supervised = len(self.losses) - 1

        backbone_keys = (
            [selector_backbones]
            if isinstance(selector_backbones, RegistrationKey)
            else selector_backbones
        )
        selector_keys = (
            [selectors] if isinstance(selectors, RegistrationKey) else selectors
        )
        if not backbone_keys or len(backbone_keys) != len(selector_keys):
            raise ValueError("SPP requires one selector backbone per selector")

        self.selector_backbones = th.nn.ModuleList(
            Registry.from_keys(backbone_keys, expected_type=SPPBackbone)
        )
        self.selectors = th.nn.ModuleList(
            Registry.from_key(
                key,
                expected_type=SPPSelector,
                input_size=backbone.output_size,
            )
            for key, backbone in zip(selector_keys, self.selector_backbones)
        )

        self.predictor_backbone = (
            Registry.from_key(predictor_backbone, expected_type=SPPBackbone)
            if predictor_backbone is not None
            else self.selector_backbones[0]
        )
        self.predictor = Registry.from_key(
            predictor,
            expected_type=SPPPredictor,
            input_size=self.predictor_backbone.output_size,
        )
        self.aggregator = (
            Registry.from_key(aggregator, expected_type=SPPAggregator)
            if aggregator is not None
            else SPPFirstAggregator()
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        self.temperature = temperature
        if select_over not in ("word", "subtoken"):
            raise ValueError("select_over must be 'word' or 'subtoken'")
        # A word is what the corpus annotates, what a sparsity target is a
        # fraction of, and what an export shows -- and it is the same unit
        # whichever backbone read the text. `subtoken` is the older behaviour,
        # kept so the difference can be measured rather than argued about.
        self.select_over = select_over

    @property
    def selector_backbone(self) -> SPPBackbone:
        return self.selector_backbones[0]

    def selection_valid(self, data: InputData) -> th.Tensor:
        """Which positions of the selection axis hold something selectable.

        The word axis says so directly. On the subtoken axis a special token
        is not a word and is never a choice, so it is excluded here even
        though the encoder always attends over it.
        """
        if self.select_over == "word" or data.word_ids is None:
            return data.mask
        return (data.word_ids >= 0).to(data.mask.dtype)

    def selection_truth(self, data: InputData) -> th.Tensor:
        """The annotation on the selection axis, ``-1`` where there is none."""
        if self.select_over == "word" or data.word_ids is None:
            return data.highlight_true
        index = data.word_ids.clamp_min(0)
        spread = data.highlight_true.gather(1, index)
        return th.where(data.word_ids >= 0, spread, th.full_like(spread, -1))

    def namespace(
        self, input_data: InputData, output_data, **extra: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """The batch as losses and metrics see it, on the selection axis.

        ``mask`` and ``highlight_true`` arrive on the word axis, which is the
        selection axis unless a model was told otherwise. A model selecting
        over subtokens gets them spread onto that axis instead, so a loss
        binds the same field name either way and always scores the unit the
        selection was made in.
        """
        values = super().namespace(input_data, output_data, **extra)
        if self.select_over == "subtoken":
            values["mask"] = self.selection_valid(input_data)
            values["highlight_true"] = self.selection_truth(input_data)
        return values

    def to_words(self, states: th.Tensor, data: InputData) -> th.Tensor:
        """Average each word's subtoken states into one state for the word.

        A selection is made over these, so it is made over the unit a person
        reads and the corpus annotates. Selecting over subtokens instead lets
        a model keep ``un`` and drop ``##fair``, which the export then reports
        as the word ``unfair`` -- a highlight that is not what the predictor
        read, in a library whose whole claim is that it is.

        Pooling happens *after* the encoder, never before: the backbone still
        attends over its own subtokens and stays on the distribution it was
        pretrained on. For a vocabulary tokenizer the axes coincide and this
        is the identity.
        """
        # No word ids means the axes already coincide -- a vocabulary
        # tokenizer, or a batch assembled by hand in a test.
        if self.select_over == "subtoken" or data.word_ids is None:
            return states
        word_ids, width = data.word_ids, data.mask.shape[1]
        # -1 marks a special token or padding; folding those into slot 0 would
        # mix `[CLS]` into the first word, so they are sent to a slot past the
        # end and dropped with the slice.
        index = word_ids.clamp_min(-1).masked_fill(word_ids < 0, width)
        pooled = states.new_zeros((states.shape[0], width + 1, states.shape[2]))
        pooled.scatter_reduce_(
            dim=1,
            index=index.unsqueeze(-1).expand_as(states),
            src=states,
            reduce="mean",
            include_self=False,
        )
        return pooled[:, :width]

    def to_subtokens(self, selection: th.Tensor, data: InputData) -> th.Tensor:
        """Spread a word's decision back over the subtokens that spell it.

        The predictor reads subtokens, so a word-level selection has to become
        one. Every position of a selected word is kept and every position of a
        dropped word is dropped, which is what makes the exported highlight
        exactly the predictor's input rather than an approximation of it.

        A special token belongs to no word and is always kept: it is not
        content, so it is never a choice, but the encoder was pretrained
        reading it.
        """
        if self.select_over == "subtoken" or data.word_ids is None:
            return selection
        word_ids = data.word_ids
        index = word_ids.clamp_min(0)
        spread = selection.gather(1, index)
        special = (word_ids < 0) & data.attention().bool()
        return th.where(special, th.ones_like(spread), spread * (word_ids >= 0))

    def load_embeddings(self, matrix: th.Tensor) -> None:
        """Hand the same pretrained table to every backbone.

        Selector and predictor read the same ids, so they read the same table;
        a backbone that shares weights with another is only loaded once.
        """
        for backbone in {
            id(backbone): backbone
            for backbone in (*self.selector_backbones, self.predictor_backbone)
        }.values():
            backbone.load_embeddings(matrix)

    def select_activation(self, highlight_logits: th.Tensor) -> th.Tensor:
        if self.training:
            return th.nn.functional.gumbel_softmax(
                highlight_logits, tau=self.temperature, hard=True, dim=-1
            )[..., 1]
        return highlight_logits.argmax(dim=-1).to(highlight_logits.dtype)

    def select(
        self,
        data: InputData,
        selector: SPPSelector,
        backbone: SPPBackbone,
    ) -> Tuple[th.Tensor, th.Tensor]:
        states = self.to_words(backbone.encode(data.features, data.attention()), data)
        highlight_logits = selector(states)
        highlight_mask = self.select_activation(highlight_logits)
        valid = self.selection_valid(data).bool()
        highlight_mask = highlight_mask * valid.to(highlight_mask.dtype)

        # A sample whose selector marks no valid token would leave the
        # predictor with an empty input, so the highest-scoring valid token is
        # selected instead. The fallback is straight-through, keeping the
        # gradient path to the selector open. GenSPP overrides this: its search
        # scores empty selections rather than repairing them.
        needs_fallback = valid.any(dim=1) & ~highlight_mask.bool().any(dim=1)
        if needs_fallback.any():
            scores = th.softmax(highlight_logits / self.temperature, dim=-1)[..., 1]
            scores = scores * valid.to(scores.dtype)
            fallback_hard = th.nn.functional.one_hot(
                scores.masked_fill(~valid, -th.inf).argmax(dim=1),
                num_classes=scores.shape[1],
            ).to(scores.dtype)
            fallback = fallback_hard + scores - scores.detach()
            highlight_mask = th.where(
                needs_fallback.unsqueeze(1), fallback, highlight_mask
            )

        return highlight_logits, highlight_mask

    def predict(self, data: InputData, highlight_mask: th.Tensor) -> th.Tensor:
        selection = self.selection_valid(data).to(highlight_mask.dtype) * highlight_mask
        # Onto the axis the encoder reads, where a special token is always
        # attended and a dropped word is gone in every piece of itself.
        prediction_mask = self.to_subtokens(selection, data)
        attention = data.attention().to(prediction_mask.dtype)
        states = self.predictor_backbone.encode(
            data.features, attention, selection_mask=prediction_mask
        )
        pooled = self.predictor_backbone.pool(states, attention * prediction_mask)
        return self.predictor(pooled)

    def predict_full(self, data: InputData) -> th.Tensor:
        """Class logits from the whole input, with nothing selected away.

        Not what a select-then-predict model does in use: its predictor reads
        the highlight and only the highlight. MCD trains this pass on purpose,
        and the faithfulness terms need it as their reference point.
        """
        return self.predict(
            data=data, highlight_mask=th.ones_like(self.selection_valid(data))
        )

    def faithfulness(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Dict[str, th.Tensor]:
        """Per-sample sufficiency and comprehensiveness.

        Scored on the head the aggregator keeps, which is the head every
        reported metric scores, and against the class that head predicts --
        see :mod:`pyhighlights.components.faithfulness` for why ``y_hat``
        comes from the highlight rather than from the full input.

        Two extra predictor passes: the full input, and the input with the
        highlight removed. The highlight pass is the model's own output and is
        read off ``output_data`` rather than recomputed. A highlight covering
        every valid token leaves the complement empty, which the backbones
        pool to zeros -- an honest measurement of a model that kept
        everything, not a case to repair.
        """
        head = self.aggregator(output_data)
        valid = self.selection_valid(input_data).to(head.highlight_mask.dtype)
        highlight = head.highlight_mask * valid

        def probability(logits: th.Tensor, of: th.Tensor) -> th.Tensor:
            return th.softmax(logits, dim=-1).gather(1, of.unsqueeze(1)).squeeze(1)

        predicted = head.class_logits.argmax(dim=-1)
        on_highlight = probability(head.class_logits, predicted)
        on_full = probability(self.predict_full(input_data), predicted)
        on_complement = probability(
            self.predict(data=input_data, highlight_mask=valid * (1 - highlight)),
            predicted,
        )
        return {
            "sufficiency": on_full - on_highlight,
            "comprehensiveness": on_full - on_complement,
        }

    def forward(self, data: InputData) -> SPPOutput:
        highlight_logits = []
        highlight_masks = []
        class_logits = []

        for backbone, selector in zip(self.selector_backbones, self.selectors):
            head_logits, head_mask = self.select(
                data=data, selector=selector, backbone=backbone
            )
            highlight_logits.append(head_logits)
            highlight_masks.append(head_mask)
            class_logits.append(self.predict(data=data, highlight_mask=head_mask))

        return SPPOutput(
            class_logits=th.stack(class_logits, dim=1),
            highlight_logits=th.stack(highlight_logits, dim=1),
            highlight_mask=th.stack(highlight_masks, dim=1),
        )

    def head_namespace(
        self, input_data: InputData, output_data: SPPOutput, **extra: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """Namespace of the first head, with the head dimension dropped."""
        return self.namespace(input_data, next(output_data.unbind(dim=1)), **extra)

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: SPPOutput
    ):
        super().update_metrics(
            split=split,
            input_data=input_data,
            output_data=self.aggregator(output_data),
        )

    def compute_loss(
        self,
        input_data: InputData,
        output_data: SPPOutput,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total_loss = output_data.class_logits.new_zeros(())
        losses: Dict[str, th.Tensor] = {}

        for index, head_output in enumerate(output_data.unbind(dim=1)):
            # There is one annotation, so it guides one head: the one the
            # aggregator keeps and every reported metric scores. Guiding the
            # rest towards the same tokens would undo what a model with several
            # generators has them for.
            head_losses = [
                loss
                for position, loss in enumerate(self.losses)
                if index == 0 or position != self.supervised
            ]
            head_loss, computed = compute_losses(
                head_losses, self.namespace(input_data, head_output)
            )
            total_loss = total_loss + head_loss
            for name, value in computed.items():
                losses[name] = losses.get(name, value.new_zeros(())) + value

        return total_loss, losses
