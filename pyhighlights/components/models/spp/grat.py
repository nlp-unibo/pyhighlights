from typing import List

import torch as th
from torch.nn.functional import gumbel_softmax

from pyhighlights.components.models.spp.base import (
    SPP,
    SPPInputData,
    SPPEmbedder,
    SPPEncoder,
    SPPPredictor,
)
from pyhighlights.components.models.spp.implementations import (
    GRUEmbedder,
    GRUEncoder,
    GRUPredictor,
    GRUSelector,
    TransformerEmbedder,
    TransformerEncoder,
    TransformerPredictor,
    TransformerSelector,
)
from pyhighlights.models.layers import AttentionPooling


class GRAT(SPP):
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature

    def select_activation(
        self,
        highlight_logits: th.Tensor,
    ) -> th.Tensor:
        # highlight_logits: [bs, F, 2]

        # [bs, F]
        return gumbel_softmax(logits=highlight_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]


class GRATGuiderConverter(th.nn.Module):
    def __init__(
        self, embedding_dim: int, encoder_output_dim: int, hidden_sizes: List[int]
    ):
        super().__init__()

        self.guider = th.nn.Sequential()

        hidden_sizes.insert(0, embedding_dim)
        hidden_sizes.append(encoder_output_dim)
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.guider.append(th.nn.Linear(input_size, hidden_size))

    def forward(self, embeddings: th.Tensor) -> th.Tensor:
        # [bs, F, d]

        # [bs, F, d_2]
        return self.guider(embeddings)


class GRATGuiderAttention(th.nn.Module):
    def __init__(self, encoder_output_dim: int, hidden_sizes: List[int]):
        super().__init__()

        self.attention = th.nn.Sequential()

        hidden_sizes.insert(0, encoder_output_dim)
        hidden_sizes.append(1)
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.attention.append(th.nn.Linear(input_size, hidden_size))
            self.attention.append(th.nn.GELU())

    def forward(self, encodings: th.Tensor) -> th.Tensor:
        # [bs, F, d]

        # [bs, F, 1]
        return self.attention(encodings)


class GRATGuiderProjector(th.nn.Module):
    def __init__(self, encoder_output_dim: int, hidden_sizes: List[int]):
        super().__init__()

        self.projector = th.nn.Sequential()

        hidden_sizes.insert(0, encoder_output_dim)
        hidden_sizes.append(encoder_output_dim)
        for input_size, hidden_size in zip(hidden_sizes[:-1], hidden_sizes[1:]):
            self.projector.append(th.nn.Linear(input_size, hidden_size))
            self.projector.append(th.nn.GELU())

    def forward(self, encodings: th.Tensor, attention_weights: th.Tensor) -> th.Tensor:
        # encodings:            [bs, F, d]
        # attention_weights:    [bs, F, 1]

        # [bs, F, 1]
        return self.projector(encodings * attention_weights)


class GRATGuider(th.nn.Module):
    def __init__(
        self,
        embedder: SPPEmbedder,
        converter: GRATGuiderConverter,
        encoder: SPPEncoder,
        attention: GRATGuiderAttention,
        projector: GRATGuiderProjector,
        pooler: AttentionPooling,
        predictor: SPPPredictor,
        noise_sigma=1.0,
        temperature=1.0,
    ):
        super().__init__()

        self.embedder = embedder
        self.converter = converter
        self.encoder = encoder
        self.attention = attention
        self.projector = projector
        self.pooler = pooler
        self.predictor = predictor

        self.temperature = temperature
        self.noise_sigma = noise_sigma

    def forward(self, data: SPPInputData):
        # data.features:    [bs, F]
        # data.mask:        [bs, F]

        bool_mask = data.mask.to(th.bool)

        # [bs, F, d]
        embeddings = self.embedder.forward(features=data.features, mask=data.mask)

        # [bs, F, d_2]
        input_states = self.converter.forward(embeddings=embeddings)

        # [bs, F, d_2]
        encodings = self.encoder(embeddings=embeddings)
        encodings = self.layer_norm(encodings + input_states)

        # [bs, F, 1]
        highlight_mask = self.attention.forward(encodings=encodings)
        highlight_mask = highlight_mask.masked_fill_(
            ~bool_mask[:, :, None], th.finfo(th.float).min
        )

        if self.training:
            # [bs, F, 1]
            attention_noises = th.normal(
                0,
                self.noise_sigma,
                highlight_mask.size(),
                device=highlight_mask.device,
                dtype=highlight_mask.dtype,
            )
            attention_noises = th.abs(attention_noises).masked_fill_(
                ~bool_mask[:, :, None], th.finfo(th.float).min
            )
            highlight_mask += attention_noises

        # [bs, F, 1]
        highlight_mask = th.nn.functional.softmax(highlight_mask, dim=1)

        # [bs, F, d_2]
        projected = self.projector.forward(
            encodings=encodings, attention_weights=highlight_mask
        )

        final_states = self.pooler.forward(hidden_states=projected, mask=data.mask)
        predictor_logits = self.predictor.forward(encodings=final_states)

        return highlight_mask, predictor_logits


# ---------------------------------------------------------------------------
# GRU-backed GRAT
# ---------------------------------------------------------------------------


class GRUGRAT(GRAT):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_input_size: int,
        encoder_hidden_size: int,
        selector_hidden_sizes: List[int],
        predictor_hidden_sizes: List[int],
        num_classes: int,
        embedding_matrix: th.Tensor | None = None,
        freeze_embeddings: bool = False,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate=0.0,
        temperature: float = 1.0,
    ):
        embedder = GRUEmbedder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=freeze_embeddings,
        )

        encoder = GRUEncoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
        )

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selectors=GRUSelector(hidden_sizes=selector_hidden_sizes),
            predictor=GRUPredictor(
                hidden_sizes=predictor_hidden_sizes, num_classes=num_classes
            ),
            temperature=temperature,
        )


class GRUGRATGuider(GRATGuider):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_input_size: int,
        encoder_hidden_size: int,
        converter_hidden_sizes: List[int],
        attention_hidden_sizes: List[int],
        projector_hidden_sizes: List[int],
        predictor_hidden_sizes: List[int],
        num_classes: int,
        embedding_matrix: th.Tensor | None = None,
        freeze_embeddings: bool = False,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate=0.0,
        noise_sigma=1.0,
        temperature=1.0,
    ):
        embedder = GRUEmbedder(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            embedding_matrix=embedding_matrix,
            freeze_embeddings=freeze_embeddings,
        )
        encoder_output_dim = (
            encoder_hidden_size * 2 if bidirectional else encoder_hidden_size
        )

        converter = GRATGuiderConverter(
            embedding_dim=embedding_dim,
            encoder_output_dim=encoder_output_dim,
            hidden_sizes=converter_hidden_sizes,
        )

        encoder = GRUEncoder(
            input_size=encoder_input_size,
            hidden_size=encoder_hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            dropout_rate=dropout_rate,
        )

        attention = GRATGuiderAttention(
            encoder_output_dim=encoder_output_dim, hidden_sizes=attention_hidden_sizes
        )

        projector = GRATGuiderProjector(
            encoder_output_dim=encoder_output_dim, hidden_sizes=projector_hidden_sizes
        )

        pooler = AttentionPooling(
            in_dim=encoder_output_dim, hidden_size=encoder_output_dim
        )

        predictor = GRUPredictor(
            hidden_sizes=predictor_hidden_sizes, num_classes=num_classes
        )

        super().__init__(
            embedder=embedder,
            converter=converter,
            encoder=encoder,
            attention=attention,
            projector=projector,
            pooler=pooler,
            predictor=predictor,
            noise_sigma=noise_sigma,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Transformer-backed GRAT
# ---------------------------------------------------------------------------


class TransformerGRAT(GRAT):
    def __init__(
        self,
        pretrained_model_card: str,
        num_features: int,
        selector_hidden_sizes: List[int],
        predictor_hidden_sizes: List[int],
        num_classes: int,
        freeze_transformer: bool = False,
        temperature: float = 1.0,
    ):
        embedder = TransformerEmbedder(
            pretrained_model_card=pretrained_model_card,
            num_features=num_features,
            freeze_transformer=freeze_transformer,
        )

        encoder = TransformerEncoder()

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selectors=TransformerSelector(hidden_sizes=selector_hidden_sizes),
            predictor=TransformerPredictor(
                hidden_sizes=predictor_hidden_sizes, num_classes=num_classes
            ),
            temperature=temperature,
        )
