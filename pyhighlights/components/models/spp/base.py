import abc
from typing import List, Tuple, Union

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, Model, OutputData, Split

# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------


class SPPEmbedder(th.nn.Module, abc.ABC):
    """Encodes raw input into a representation for the selector/predictor."""

    @abc.abstractmethod
    def forward(self, features: th.Tensor, mask: th.Tensor) -> th.Tensor: ...


class SPPEncoder(th.nn.Module, abc.ABC):
    """Encodes raw input into a representation for the selector/predictor."""

    @abc.abstractmethod
    def encode_features(self, embeddings: th.Tensor) -> th.Tensor: ...

    @abc.abstractmethod
    def pool_encodings(self, encodings: th.Tensor, mask: th.Tensor) -> th.Tensor: ...


class SPPSelector(th.nn.Module, abc.ABC):
    """Produces the rationale (e.g. per-token selection scores)."""

    @abc.abstractmethod
    def forward(self, encodings: th.Tensor) -> th.Tensor: ...


class SPPPredictor(th.nn.Module, abc.ABC):
    """Produces the final prediction from the selected rationale."""

    @abc.abstractmethod
    def forward(self, encodings: th.Tensor) -> th.Tensor: ...


class SPPAggregator(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, output_data: OutputData) -> OutputData: ...


class SPPFirstAggregator(SPPAggregator):
    def forward(self, output_data: OutputData) -> OutputData:
        output_data.highlight_logits = output_data.highlight_logits[:, 0, :, :]
        output_data.highlight_pred = output_data.highlight_pred[:, 0, :]
        output_data.y_pred = output_data.y_pred[:, 0, :]
        return output_data


# ---------------------------------------------------------------------------
# Base SPP model
# ---------------------------------------------------------------------------


class SPP(Model, abc.ABC):
    # TODO: add skew setup
    def __init__(
        self,
        selector_embedder: RegistrationKey[SPPEmbedder],
        selector_encoder: RegistrationKey[SPPEncoder],
        selectors: Union[
            RegistrationKey[SPPSelector], List[RegistrationKey[SPPSelector]]
        ],
        predictor: RegistrationKey[SPPPredictor],
        predictor_embedder: RegistrationKey[SPPEmbedder] | None = None,
        predictor_encoder: RegistrationKey[SPPEncoder] | None = None,
        aggregator: RegistrationKey[SPPAggregator] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.selector_embedder = Registry.from_key(selector_embedder)
        self.selector_encoder = Registry.from_key(selector_encoder)

        if isinstance(selectors, RegistrationKey):
            selectors = [selectors]
        self.selectors = th.nn.ModuleList([Registry.from_key(key) for key in selectors])

        if predictor_embedder is not None:
            self.predictor_embedder = Registry.from_key(predictor_embedder)
        else:
            self.predictor_embedder = selector_embedder

        if predictor_encoder is not None:
            self.predictor_encoder = Registry.from_key(predictor_encoder)
        else:
            self.predictor_encoder = selector_encoder
        self.predictor = Registry.from_key(predictor)

        self.aggregator = aggregator or SPPFirstAggregator()

    def encode_features(
        self, embeddings: th.Tensor, mask: th.Tensor, encoder: SPPEncoder
    ) -> th.Tensor:
        # embeddings:   [bs, F, d]
        # mask:         [bs, F]

        # [bs, F, d_1]
        encodings = encoder.encode_features(embeddings=embeddings)
        encodings *= mask[:, :, None]
        return encodings

    def pool_encodings(
        self, encodings: th.Tensor, mask: th.Tensor, encoder: SPPEncoder
    ) -> th.Tensor:
        # encodings:    [bs, F, d_2]

        # [bs, d_3]
        encodings = encoder.pool_encodings(encodings=encodings, mask=mask)
        return encodings

    def select(
        self, data: InputData, selector: SPPSelector
    ) -> Tuple[th.Tensor, th.Tensor]:

        # [bs, F, d]
        embeddings = self.selector_embedder.forward(
            features=data.features, mask=data.mask
        )

        # [bs, F, d_1]
        encodings = self.encode_features(
            embeddings=embeddings, mask=data.mask, encoder=self.selector_encoder
        )

        # [bs, F, 2]
        highlight_logits = selector.forward(encodings=encodings)

        # [bs, F]
        highlight_pred = self.select_activation(highlight_logits=highlight_logits)

        return highlight_logits, highlight_pred

    def select_activation(
        self,
        highlight_logits: th.Tensor,
    ) -> th.Tensor:
        # highlight_logits: [bs, F, 2]

        # [bs, F]
        return th.nn.functional.softmax(highlight_logits, dim=-1)[:, :, 1]

    def predict(self, data: InputData, highlight_pred: th.Tensor):
        # data.features:        [bs, F]
        # data.mask:            [bs, F]
        # highlight_pred:       [bs, F]

        # [bs, F, d]
        embeddings = self.selector_embedder.forward(
            features=data.features, mask=highlight_pred
        )

        # [bs, F, d_2]
        encodings = self.encode_features(
            embeddings=embeddings, mask=highlight_pred, encoder=self.predictor_encoder
        )

        # [bs, F, d_3]
        pooled_encodings = self.pool_encodings(
            encodings=encodings, mask=highlight_pred, encoder=self.predictor_encoder
        )

        # [bs, C]
        predictor_logits = self.predictor.forward(encodings=pooled_encodings)

        return predictor_logits

    def forward(self, data: InputData) -> OutputData:
        # data.features:    [bs, F]
        # data.mask:        [bs, F]
        # data.sample_ids:  [bs,]

        # [bs, S, F, 2] where S = len(self.selectors)
        highlights_logits = []

        # [bs, S, F]
        highlight_preds = []

        # [bs, S, C]
        predictors_logits = []

        for selector in self.selectors:
            # [bs, F, 2], [bs, F]
            highlight_logits, highlight_pred = self.select(data=data, selector=selector)
            highlights_logits.append(highlight_logits)
            highlight_preds.append(highlight_pred)

            # [bs, C] where C = no. of classes
            predictor_logits = self.predict(data=data, highlight_pred=highlight_pred)
            predictors_logits.append(predictor_logits)

        highlights_logits = th.stack(highlights_logits, dim=1)
        highlight_preds = th.stack(highlight_preds, dim=1)
        predictors_logits = th.stack(predictors_logits, dim=1)

        return OutputData(
            highlight_logits=highlights_logits,
            y_pred=predictors_logits,
            highlight_pred=highlight_preds,
        )

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: OutputData
    ):
        output_data = self.aggregator.forward(output_data=output_data)
        super().update_metrics(
            split=split, input_data=input_data, output_data=output_data
        )

    # TODO: override compute_loss to iterate over S
