from typing import List, Tuple

import torch as th
from torch.nn.functional import gumbel_softmax

from pyhighlights.components.models.spp.base import SPP, SPPInputData
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


class MGR(SPP):
    def __init__(self, temperature: float = 1.0, **kwargs):
        super().__init__(**kwargs)

        self.temperature = temperature

    def select_activation(
        self,
        selector_logits: th.Tensor,
    ) -> th.Tensor:
        # selector_logits: [bs, F, 2]

        # [bs, F]
        return gumbel_softmax(logits=selector_logits, tau=self.temperature, hard=True)[
            :, :, 1
        ]

    def forward_one_head(
        self, data: SPPInputData, selector_idx: int
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, SPPInputData]:
        # data.features:    [bs, F]
        # data.mask:        [bs, F]
        # data.sample_ids:  [bs,]

        # [bs, F, 2], [bs, F]
        selector_logits, highlight_mask = self.select(
            data=data, selector=self.selectors[selector_idx]
        )

        # [bs, C]
        predictor_logits = self.predict(data=data, highlight_mask=highlight_mask)

        return selector_logits, predictor_logits, highlight_mask, data


# ---------------------------------------------------------------------------
# GRU-backed MGR
# ---------------------------------------------------------------------------


class GRUMGR(MGR):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_input_size: int,
        encoder_hidden_size: int,
        selector_hidden_sizes: List[int],
        predictor_hidden_sizes: List[int],
        num_classes: int,
        num_selectors=1,
        embedding_matrix: th.Tensor | None = None,
        freeze_embeddings: bool = False,
        num_layers: int = 1,
        bidirectional: bool = True,
        dropout_rate=0.0,
        **kwargs,
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

        selectors = [
            GRUSelector(hidden_sizes=selector_hidden_sizes)
            for _ in range(num_selectors)
        ]

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selector=selectors,
            predictor=GRUPredictor(
                hidden_sizes=predictor_hidden_sizes, num_classes=num_classes
            ),
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Transformer-backed MGR
# ---------------------------------------------------------------------------


class TransformerMGR(MGR):
    def __init__(
        self,
        pretrained_model_card: str,
        num_features: int,
        selector_hidden_sizes: List[int],
        predictor_hidden_sizes: List[int],
        num_classes: int,
        num_selectors: int = 1,
        freeze_transformer: bool = False,
        **kwargs,
    ):
        embedder = TransformerEmbedder(
            pretrained_model_card=pretrained_model_card,
            num_features=num_features,
            freeze_transformer=freeze_transformer,
        )

        encoder = TransformerEncoder()

        selectors = [
            TransformerSelector(hidden_sizes=selector_hidden_sizes)
            for _ in range(num_selectors)
        ]

        super().__init__(
            selector_embedder=embedder,
            predictor_embedder=embedder,
            selector_encoder=encoder,
            predictor_encoder=encoder,
            selectors=selectors,
            predictor=TransformerPredictor(
                hidden_sizes=predictor_hidden_sizes, num_classes=num_classes
            ),
            **kwargs,
        )
