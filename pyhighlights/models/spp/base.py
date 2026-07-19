import abc
from typing import Tuple, Union, Sequence

import torch as th


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

class SPPData:

    def __init__(
            self,
            features: th.Tensor,
            mask: th.Tensor,
            sample_ids: th.Tensor
    ):
        self.features = features
        self.mask = mask
        self.sample_ids = sample_ids


# ---------------------------------------------------------------------------
# Component interfaces
# ---------------------------------------------------------------------------

class SPPEmbedder(th.nn.Module, abc.ABC):
    """Encodes raw input into a representation for the selector/predictor."""

    @abc.abstractmethod
    def forward(
            self,
            features: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        ...


class SPPEncoder(th.nn.Module, abc.ABC):
    """Encodes raw input into a representation for the selector/predictor."""

    @abc.abstractmethod
    def encode_features(
            self,
            embeddings: th.Tensor
    ) -> th.Tensor:
        ...

    @abc.abstractmethod
    def pool_encodings(
            self,
            encodings: th.Tensor,
            mask: th.Tensor
    ) -> th.Tensor:
        ...


class SPPSelector(th.nn.Module, abc.ABC):
    """Produces the rationale (e.g. per-token selection scores)."""

    @abc.abstractmethod
    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        ...


class SPPPredictor(th.nn.Module, abc.ABC):
    """Produces the final prediction from the selected rationale."""

    @abc.abstractmethod
    def forward(
            self,
            encodings: th.Tensor
    ) -> th.Tensor:
        ...


# ---------------------------------------------------------------------------
# Base SPP model
# ---------------------------------------------------------------------------


class SPP(th.nn.Module):

    def __init__(
            self,
            selector_embedder: SPPEmbedder,
            predictor_embedder: SPPEmbedder,
            selector_encoder: SPPEncoder,
            predictor_encoder: SPPEncoder,
            selectors: Union[SPPSelector, Sequence[SPPSelector]],
            predictor: SPPPredictor
    ):
        super().__init__()

        self.selector_embedder = selector_embedder
        self.selector_encoder = selector_encoder

        if isinstance(selectors, SPPSelector):
            selectors = [selectors]
        self.selectors = th.nn.ModuleList(selectors)

        self.predictor_embedder = predictor_embedder
        self.predictor_encoder = predictor_encoder
        self.predictor = predictor

    def encode_features(
            self,
            embeddings: th.Tensor,
            mask: th.Tensor,
            encoder: SPPEncoder
    ) -> th.Tensor:
        # embeddings:   [bs, F, d]
        # mask:         [bs, F]

        # [bs, F, d_1]
        encodings = encoder.encode_features(embeddings=embeddings)
        encodings *= mask[:, :, None]
        return encodings

    def pool_encodings(
            self,
            encodings: th.Tensor,
            mask: th.Tensor,
            encoder: SPPEncoder
    ) -> th.Tensor:
        # encodings:    [bs, F, d_2]

        # [bs, d_3]
        encodings = encoder.pool_encodings(encodings=encodings, mask=mask)
        return encodings

    def select(
            self,
            data: SPPData,
            selector: SPPSelector
    ) -> Tuple[th.Tensor, th.Tensor]:

        # [bs, F, d]
        embeddings = self.selector_embedder.forward(features=data.features, mask=data.mask)

        # [bs, F, d_1]
        encodings = self.encode_features(embeddings=embeddings,
                                         mask=data.mask,
                                         encoder=self.selector_encoder)

        # [bs, F, 2]
        selector_logits = selector.forward(encodings=encodings)

        # [bs, F]
        highlight_mask = self.select_activation(selector_logits=selector_logits)

        return selector_logits, highlight_mask

    def select_activation(
            self,
            selector_logits: th.Tensor,
    ) -> th.Tensor:
        # selector_logits: [bs, F, 2]

        # [bs, F]
        return th.nn.functional.softmax(selector_logits, dim=-1)[:, :, 1]

    def predict(
            self,
            data: SPPData,
            highlight_mask: th.Tensor
    ):
        # data.features:        [bs, F]
        # data.mask:            [bs, F]
        # highlight_mask:       [bs, F]

        # [bs, F, d]
        embeddings = self.selector_embedder.forward(features=data.features,
                                                    mask=highlight_mask)

        # [bs, F, d_2]
        encodings = self.encode_features(embeddings=embeddings,
                                         mask=highlight_mask,
                                         encoder=self.predictor_encoder)

        # [bs, F, d_3]
        pooled_encodings = self.pool_encodings(encodings=encodings,
                                               mask=highlight_mask,
                                               encoder=self.predictor_encoder)

        # [bs, C]
        predictor_logits = self.predictor.forward(encodings=pooled_encodings)

        return predictor_logits

    def forward(
            self,
            data: SPPData
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor, SPPData]:
        # data.features:    [bs, F]
        # data.mask:        [bs, F]
        # data.sample_ids:  [bs,]

        # [bs, S, F, 2] where S = len(self.selectors)
        selectors_logits = []

        # [bs, S, F]
        highlight_masks = []

        # [bs, S, C]
        predictors_logits = []

        for selector in self.selectors:
            # [bs, F, 2], [bs, F]
            selector_logits, highlight_mask = self.select(data=data, selector=selector)
            selectors_logits.append(selector_logits)
            highlight_masks.append(highlight_mask)

            # [bs, C] where C = no. of classes
            predictor_logits = self.predict(data=data, highlight_mask=highlight_mask)
            predictors_logits.append(predictor_logits)

        selectors_logits = th.stack(selectors_logits, dim=1)
        highlight_masks = th.stack(highlight_masks, dim=1)
        predictors_logits = th.stack(predictors_logits, dim=1)

        return selectors_logits, predictors_logits, highlight_masks, data
