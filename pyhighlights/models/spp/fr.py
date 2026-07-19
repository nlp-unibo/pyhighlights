from typing import List

import torch as th
from torch.nn.functional import gumbel_softmax
from transformers import AutoModel

from pyhighlights.models.spp.base import SPP, SPPEmbedder, SPPEncoder, SPPSelector, SPPPredictor


class FR(SPP):

    def __init__(
            self,
            temperature: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.temperature = temperature

        if self.selector_embedder != self.predictor_embedder:
            raise AttributeError(
                'Folded Rationalization (FR) requires a shared embedder between selector and predictor')

        if self.selector_encoder != self.predictor_encoder:
            raise AttributeError('Folded Rationalization (FR) requires a shared encoder between selector and predictor')

    def select_activation(
            self,
            selector_logits: th.Tensor,
    ) -> th.Tensor:
        # selector_logits: [bs, F, 2]

        # [bs, F]
        return gumbel_softmax(logits=selector_logits,
                              tau=self.temperature,
                              hard=True)[:, :, 1]


# ---------------------------------------------------------------------------
# GRU-backed FR
# ---------------------------------------------------------------------------

class GRUEmbedder(SPPEmbedder):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            embedding_matrix: th.Tensor | None = None,
            freeze_embeddings: bool = False
    ):
        super().__init__()

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

    def forward(
            self,
            features: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        # features:     [bs, F]
        # mask:         [bs, F]

        # [bs, F, embedding_dim]
        return self.embedding(features) * mask[:, :, None]


class GRUEncoder(SPPEncoder):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            bidirectional: bool = True,
            dropout_rate=0.0,
    ):
        super().__init__()

        self.encoder = th.nn.GRU(input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 batch_first=True,
                                 bidirectional=bidirectional)
        self.dropout = th.nn.Dropout(p=dropout_rate)

        self.layer_norm = th.nn.LayerNorm(hidden_size * 2 if bidirectional else hidden_size)

    def encode_features(
            self,
            embeddings: th.Tensor
    ) -> th.Tensor:
        # embeddings:   [bs, F, input_size]

        # [bs, F, hidden_size] or [bs, F, hidden_size *2] if bidirectional is True
        encodings, _ = self.encoder(embeddings)
        encodings = self.layer_norm(encodings)
        encodings = self.dropout(encodings)
        return encodings

    def pool_encodings(
            self,
            encodings: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        # encodings:    [bs, F, hidden_size]
        # mask:         [bs, F]

        encodings = encodings * mask[:, :, None] + (1. - mask[:, :, None]) * (-1e6)
        encodings = th.transpose(encodings, 1, 2)

        # [bs, hidden_size]
        hl_emb, _ = th.max(encodings, dim=2)
        return hl_emb


class GRUSelector(SPPSelector):

    def __init__(
            self,
            hidden_sizes: List[int]
    ):
        super().__init__()

        self.selector = th.nn.Sequential()
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.selector.append(th.nn.Linear(input_size, hidden_size))
        self.selector.append(th.nn.Linear(hidden_sizes[-1], 2))

    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        # [bs, F, d]

        # [bs, F, 2]
        return self.selector(encodings)


class GRUPredictor(SPPPredictor):

    def __init__(
            self,
            hidden_sizes: List[int],
            C: int
    ):
        super().__init__()

        self.predictor = th.nn.Sequential()
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.predictor.append(th.nn.Linear(input_size, hidden_size))
        self.predictor.append(th.nn.Linear(hidden_sizes[-1], C))

    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        # [bs, d]

        # [bs, C]
        return self.predictor(encodings)


class GRUFR(FR):

    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int,
            encoder_input_size: int,
            encoder_hidden_size: int,
            selector_hidden_sizes: List[int],
            predictor_hidden_sizes: List[int],
            C: int,
            embedding_matrix: th.Tensor | None = None,
            freeze_embeddings: bool = False,
            num_layers: int = 1,
            bidirectional: bool = True,
            dropout_rate=0.0,
    ):
        embedder = GRUEmbedder(vocab_size=vocab_size,
                               embedding_dim=embedding_dim,
                               embedding_matrix=embedding_matrix,
                               freeze_embeddings=freeze_embeddings)

        encoder = GRUEncoder(input_size=encoder_input_size,
                             hidden_size=encoder_hidden_size,
                             num_layers=num_layers,
                             bidirectional=bidirectional,
                             dropout_rate=dropout_rate)

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selector=GRUSelector(hidden_sizes=selector_hidden_sizes),
            predictor=GRUPredictor(hidden_sizes=predictor_hidden_sizes, C=C)
        )


# ---------------------------------------------------------------------------
# Transformer-backed FR
# ---------------------------------------------------------------------------

class TransformerEmbedder(SPPEmbedder):

    def __init__(
            self,
            pretrained_model_card: str,
            num_features: int,
            freeze_transformer: bool = False,
    ):
        super().__init__()

        self.transformer = AutoModel.from_pretrained(pretrained_model_name_or_path=pretrained_model_card)
        self.transformer.resize_token_embeddings(num_features)

        self.freeze_transformer = freeze_transformer
        if freeze_transformer:
            for module in self.transformer.modules():
                for param in module.parameters():
                    param.requires_grad = False
        else:
            self.transformer.train()

    def forward(
            self,
            features: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        # features:     [bs, F]
        # mask:         [bs, F]

        # [bs, F, d]
        return self.transformer(input_ids=features, attention_mask=mask).last_hidden_state


class TransformerEncoder(SPPEncoder):

    def encode_features(
            self,
            embeddings: th.Tensor
    ) -> th.Tensor:
        # [bs, F, d]
        return embeddings

    def pool_encodings(
            self,
            encodings: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        # encodings:    [bs, F, d]
        # mask:         [bs, F]

        # [bs, d]
        pooled_encodings = (encodings * mask[:, :, None]).sum(dim=1) / mask.sum(dim=1)[:, None]
        return pooled_encodings


class TransformerSelector(SPPSelector):

    def __init__(
            self,
            hidden_sizes: List[int]
    ):
        super().__init__()

        self.selector = th.nn.Sequential()
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.selector.append(th.nn.Linear(input_size, hidden_size))
        self.selector.append(th.nn.Linear(hidden_sizes[-1], 2))

    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        # [bs, F, d]

        # [bs, F, 2]
        return self.selector(encodings)


class TransformerPredictor(SPPPredictor):

    def __init__(
            self,
            hidden_sizes: List[int],
            C: int
    ):
        super().__init__()

        self.predictor = th.nn.Sequential()
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.predictor.append(th.nn.Linear(input_size, hidden_size))
        self.predictor.append(th.nn.Linear(hidden_sizes[-1], C))

    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        # [bs, d]

        # [bs, C]
        return self.predictor(encodings)


class TransformerFR(FR):

    def __init__(
            self,
            pretrained_model_card: str,
            num_features: int,
            selector_hidden_sizes: List[int],
            C: int,
            freeze_transformer: bool = False,
    ):
        embedder = TransformerEmbedder(pretrained_model_card=pretrained_model_card,
                                       num_features=num_features,
                                       freeze_transformer=freeze_transformer)

        encoder = TransformerEncoder()

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selector=TransformerSelector(hidden_sizes=selector_hidden_sizes),
            predictor=TransformerPredictor(hidden_sizes=predictor_hidden_sizes, C=C)
        )
