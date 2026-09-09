from __future__ import annotations

from typing import List

import torch as th

from pyhighlights.components.models.spp.base import (
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)


class GRUBackbone(SPPBackbone):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        embedding_matrix: th.Tensor | None = None,
        freeze_embeddings: bool = False,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.embedding = th.nn.Embedding(vocab_size, embedding_dim)
        if embedding_matrix is not None:
            with th.no_grad():
                self.embedding.weight.copy_(embedding_matrix)
        self.embedding.weight.requires_grad_(not freeze_embeddings)

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
        """Replace the embedding table with ``matrix``, keeping it frozen or not.

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
        # The replacement lands where the old table was: a backbone already
        # moved to a device would otherwise hold a CPU table.
        self.embedding = th.nn.Embedding.from_pretrained(
            matrix.to(device=weight.device, dtype=weight.dtype),
            freeze=not weight.requires_grad,
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
        packed = th.nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths=valid.sum(dim=1).clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        states, _ = self.encoder(packed)
        states, _ = th.nn.utils.rnn.pad_packed_sequence(
            states, batch_first=True, total_length=features.shape[1]
        )
        states = self.dropout(self.layer_norm(states))
        return states * valid.unsqueeze(-1)

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        valid = mask.bool()
        pooled = states.masked_fill(~valid.unsqueeze(-1), -th.inf).amax(dim=1)
        return th.where(valid.any(dim=1, keepdim=True), pooled, th.zeros_like(pooled))


class TransformerBackbone(SPPBackbone):
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

    @property
    def output_size(self) -> int:
        return self.transformer.config.hidden_size

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


def _mlp(sizes: List[int]) -> th.nn.Sequential:
    layers: List[th.nn.Module] = []
    for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
        layers.append(th.nn.Linear(source, target))
        if index < len(sizes) - 2:
            layers.append(th.nn.GELU())
    return th.nn.Sequential(*layers)


class MLPSelector(SPPSelector):
    def __init__(self, input_size: int, hidden_sizes: List[int]):
        super().__init__()
        self.selector = _mlp([input_size, *hidden_sizes, 2])

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.selector(states)


class MLPPredictor(SPPPredictor):
    def __init__(self, input_size: int, hidden_sizes: List[int], num_classes: int):
        super().__init__()
        self.predictor = _mlp([input_size, *hidden_sizes, num_classes])

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.predictor(states)
