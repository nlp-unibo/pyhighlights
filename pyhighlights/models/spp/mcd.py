from typing import List

import torch as th
from torch.nn.functional import gumbel_softmax

from pyhighlights.models.spp.base import SPP
from pyhighlights.models.spp.implementations import (
    GRUEmbedder,
    GRUEncoder,
    GRUSelector,
    GRUPredictor,
    TransformerEmbedder,
    TransformerEncoder,
    TransformerSelector,
    TransformerPredictor
)


class MCD(SPP):

    def __init__(
            self,
            temperature: float = 1.0,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.temperature = temperature

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
# GRU-backed MCD
# ---------------------------------------------------------------------------

class GRUMCD(MCD):

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
            **kwargs
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
            selectors=GRUSelector(hidden_sizes=selector_hidden_sizes),
            predictor=GRUPredictor(hidden_sizes=predictor_hidden_sizes, num_classes=num_classes),
            **kwargs
        )


# ---------------------------------------------------------------------------
# Transformer-backed FR
# ---------------------------------------------------------------------------

class TransformerMCD(MCD):

    def __init__(
            self,
            pretrained_model_card: str,
            num_features: int,
            selector_hidden_sizes: List[int],
            predictor_hidden_sizes: List[int],
            num_classes: int,
            freeze_transformer: bool = False,
            **kwargs
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
            selectors=TransformerSelector(hidden_sizes=selector_hidden_sizes),
            predictor=TransformerPredictor(hidden_sizes=predictor_hidden_sizes, num_classes=num_classes),
            **kwargs
        )
