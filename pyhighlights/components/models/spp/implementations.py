from __future__ import annotations

from typing import List

import torch as th

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)


def recurrent_states(
    encoder: th.nn.GRU,
    layer_norm: th.nn.LayerNorm,
    dropout: th.nn.Dropout,
    inputs: th.Tensor,
    valid: th.Tensor,
) -> th.Tensor:
    """Run ``encoder`` over ``inputs``, back on the width it came in at.

    Shared by the two backbones that encode recurrently. ``GRUBackbone`` hands
    the recurrence an embedding table, and ``StackedBackbone`` a pretrained
    encoder's states. Packing keeps padding out of the recurrence.

    ``valid`` must mark a prefix of each row, with padding only at the end.
    Packing reads the first ``valid.sum()`` positions, so a row with a zero
    between ones would be packed wrongly without an error. Every mask the
    library builds pads at the end, including the compacted one. A row that
    is padding throughout gets a length of one, because a length of zero is
    not a sequence, and it is masked to zeros on the way out.
    """
    packed = th.nn.utils.rnn.pack_padded_sequence(
        inputs,
        lengths=valid.sum(dim=1).clamp_min(1).cpu(),
        batch_first=True,
        enforce_sorted=False,
    )
    states, _ = encoder(packed)
    states, _ = th.nn.utils.rnn.pad_packed_sequence(
        states, batch_first=True, total_length=inputs.shape[1]
    )
    return dropout(layer_norm(states)) * valid.unsqueeze(-1)


def max_pool(states: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """The largest value each dimension takes over the unmasked positions.

    A row with nothing unmasked pools to zeros rather than to ``-inf``. That
    row is the empty complement of a model that kept everything, and ``-inf``
    would propagate through the predictor.
    """
    valid = mask.bool()
    pooled = states.masked_fill(~valid.unsqueeze(-1), -th.inf).amax(dim=1)
    return th.where(valid.any(dim=1, keepdim=True), pooled, th.zeros_like(pooled))


class GRUBackbone(SPPBackbone):
    """A recurrent encoder over a token embedding table, max-pooled.

    A pretrained table arrives through :meth:`load_embeddings`. For the
    predictor pass, the embeddings of dropped positions are zeroed before the
    recurrence, so their content reaches no kept state.

    ``freeze_embeddings`` left at ``None`` trains a randomly initialised table
    and freezes a loaded one, since pretrained vectors are kept fixed unless a
    run asks otherwise. ``True`` freezes either table, and ``False`` trains
    either.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        freeze_embeddings: bool | None = None,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.freeze_embeddings = freeze_embeddings
        self.embedding = th.nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.requires_grad_(freeze_embeddings is not True)

        self.encoder = th.nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self._output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = th.nn.LayerNorm(self.output_size)
        self.dropout = th.nn.Dropout(dropout_rate)

    @property
    def output_size(self) -> int:
        return self._output_size

    def load_embeddings(self, matrix: th.Tensor) -> None:
        """Replace the embedding table with ``matrix``, frozen unless asked not to be.

        The table is replaced rather than copied into, because a pretrained
        vocabulary is as wide as the release covers: requiring the
        configuration to have guessed that number in advance would make
        ``vocab_size`` a value nobody can know before the corpus is read.
        """
        if matrix.shape[1] != self.embedding.embedding_dim:
            raise ValueError(
                f"embedding matrix is {matrix.shape[1]}-dimensional, but the "
                f"backbone expects {self.embedding.embedding_dim}"
            )
        weight = self.embedding.weight
        # The replacement lands where the old table was. A backbone already
        # moved to a device would otherwise hold a CPU table.
        self.embedding = th.nn.Embedding.from_pretrained(
            matrix.to(device=weight.device, dtype=weight.dtype),
            freeze=self.freeze_embeddings is not False,
        )

    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        valid = mask.bool()
        selected = mask if selection_mask is None else mask * selection_mask
        embeddings = self.embedding(features) * selected.unsqueeze(-1)
        return recurrent_states(
            self.encoder, self.layer_norm, self.dropout, embeddings, valid
        )

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        return max_pool(states, mask)


class TransformerBackbone(SPPBackbone):
    """A pretrained Hugging Face encoder, mean-pooled.

    For the predictor pass, dropped positions are removed from the attention
    mask, so no kept position attends to them. A frozen encoder stays in
    evaluation mode, so its dropout never runs and the same input always
    gives the same states.
    """

    def __init__(
        self,
        pretrained_model_card: str,
        num_features: int | None = None,
        freeze_transformer: bool = False,
    ):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as error:
            raise ImportError(
                "TransformerBackbone requires pyhighlights[transformers]"
            ) from error

        self.transformer = AutoModel.from_pretrained(pretrained_model_card)
        if num_features is not None:
            self.transformer.resize_token_embeddings(num_features)
        self.transformer.requires_grad_(not freeze_transformer)
        self.frozen = freeze_transformer

    @property
    def output_size(self) -> int:
        return self.transformer.config.hidden_size

    def train(self, mode: bool = True) -> TransformerBackbone:
        # Lightning calls this on the whole model, and freezing the weights
        # does not stop dropout. A frozen encoder is a fixed lookup, so it
        # stays in evaluation mode.
        super().train(mode)
        if self.frozen:
            self.transformer.eval()
        return self

    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        attention_mask = mask if selection_mask is None else mask * selection_mask
        states = self.transformer(
            input_ids=features, attention_mask=attention_mask
        ).last_hidden_state
        return states * mask.to(states.dtype).unsqueeze(-1)

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        float_mask = mask.to(states.dtype).unsqueeze(-1)
        return (states * float_mask).sum(dim=1) / float_mask.sum(dim=1).clamp_min(1)


def mlp(sizes: List[int]) -> th.nn.Sequential:
    """Linear layers with GELU between them and none after the last."""
    layers: List[th.nn.Module] = []
    for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
        layers.append(th.nn.Linear(source, target))
        if index < len(sizes) - 2:
            layers.append(th.nn.GELU())
    return th.nn.Sequential(*layers)


class StackedBackbone(SPPBackbone):
    """A pretrained encoder read by a recurrent one trained from scratch.

    The shape of a bidirectional GRU over a frozen embedding table, with the
    table replaced by a pretrained transformer. The transformer is the frozen
    lookup and the GRU is the encoder, so everything trainable starts from
    scratch and one learning rate serves it. A frozen transformer read by a
    linear selector would instead train a few thousand parameters, which is a
    probe rather than a select-then-predict model.

    ``freeze_transformer`` defaults to holding the weights. A trainable
    encoder underneath a trainable GRU needs two learning rates, which is what
    ``encoder_lr`` is for.
    """

    def __init__(
        self,
        pretrained_model_card: str,
        hidden_size: int = 128,
        num_features: int | None = None,
        freeze_transformer: bool = True,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.transformer = TransformerBackbone(
            pretrained_model_card=pretrained_model_card,
            num_features=num_features,
            freeze_transformer=freeze_transformer,
        )
        self.encoder = th.nn.GRU(
            input_size=self.transformer.output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self._output_size = hidden_size * (2 if bidirectional else 1)
        self.layer_norm = th.nn.LayerNorm(self._output_size)
        self.dropout = th.nn.Dropout(dropout_rate)

    @property
    def output_size(self) -> int:
        return self._output_size

    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        # The selection reaches the transformer and the GRU, and it has to
        # reach both. Masking only the transformer's attention leaves a dropped
        # subtoken with a state of its own, because the residual stream
        # carries every position's input forward. The GRU is recurrent, so
        # that state would reach every position after it, and the predictor
        # would read words the highlight excluded. `GRUBackbone` zeroes its
        # dropped embeddings for the same reason.
        states = self.transformer.encode(features, mask, selection_mask)
        if selection_mask is not None:
            states = states * selection_mask.to(states.dtype).unsqueeze(-1)
        return recurrent_states(
            self.encoder, self.layer_norm, self.dropout, states, mask.bool()
        )

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        # The GRU's pooling, since the GRU is what produced these states.
        return max_pool(states, mask)


class MLPSelector(SPPSelector):
    """A feed-forward head scoring each position as dropped or kept."""

    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()
        self.selector = mlp([input_size, *hidden_sizes, 2])

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.selector(states)

    def threshold_parameters(self) -> List[th.nn.Parameter]:
        return [self.selector[-1].bias]


class MLPPredictor(SPPPredictor):
    """A feed-forward head mapping a pooled state to class logits."""

    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super().__init__()
        self.predictor = mlp([input_size, *hidden_sizes, num_classes])

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.predictor(states)
