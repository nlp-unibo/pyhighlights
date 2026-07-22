from typing import List

import torch as th
from torch.nn.functional import gumbel_softmax

from pyhighlights.components.models.spp.base import SPP, InputData, OutputData
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
            highlight_logits: th.Tensor,
    ) -> th.Tensor:
        # highlight_logits: [bs, F, 2]

        # [bs, F]
        return gumbel_softmax(logits=highlight_logits,
                              tau=self.temperature,
                              hard=True)[:, :, 1]

    def forward_one_head(
            self, data: InputData, selector_idx: int = 0) -> OutputData:
        # data.features:    [bs, F]
        # data.mask:        [bs, F]
        # data.sample_ids:  [bs,]

        # [bs, F, 2], [bs, F]
        highlight_logits, highlight_pred = self.select(data=data,
                                                       selector=self.selectors[selector_idx])

        # [bs, C]
        predictor_logits = self.predict(data=data, highlight_pred=highlight_pred)

        # Unsqueeze to make it compatible with base class (S = 1)
        return OutputData(highlight_logits=highlight_logits.unsqueeze(dim=1),
                          highlight_pred=highlight_pred.unsqueeze(dim=1),
                          y_pred=predictor_logits.unsqueeze(dim=1))

    def validation_forward(self, data: InputData) -> OutputData:
        return self.forward_one_head(data=data, selector_idx=0)

    def test_forward(self, batch: InputData) -> OutputData:
        return self.forward_one_head(data=batch, selector_idx=0)


# TODO: move to another package (not needed)
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
