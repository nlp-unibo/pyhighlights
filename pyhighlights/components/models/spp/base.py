import abc
from typing import Sequence, Tuple, Union, Dict

import torch as th
from dataclasses import dataclass

from pyhighlights.components.models.base import Model

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class SPPInputData:
    features: th.Tensor
    mask: th.Tensor
    sample_ids: th.Tensor
    y_true: th.Tensor


@dataclass
class SPPOutputData:
    selectors_logits: th.Tensor
    predictors_logits: th.Tensor
    highlight_masks: th.Tensor

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


# ---------------------------------------------------------------------------
# Base SPP model
# ---------------------------------------------------------------------------


class SPP(Model):

    # TODO: add highlight metrics
    # TODO: add classification loss
    # TODO: add regularization losses (sparsity, contiguity)
    def __init__(
        self,
        selector_embedder: SPPEmbedder,
        predictor_embedder: SPPEmbedder,
        selector_encoder: SPPEncoder,
        predictor_encoder: SPPEncoder,
        selectors: Union[SPPSelector, Sequence[SPPSelector]],
        predictor: SPPPredictor,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.selector_embedder = selector_embedder
        self.selector_encoder = selector_encoder

        if isinstance(selectors, SPPSelector):
            selectors = [selectors]
        self.selectors = th.nn.ModuleList(selectors)

        self.predictor_embedder = predictor_embedder
        self.predictor_encoder = predictor_encoder
        self.predictor = predictor

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
        self, data: SPPInputData, selector: SPPSelector
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

    def predict(self, data: SPPInputData, highlight_mask: th.Tensor):
        # data.features:        [bs, F]
        # data.mask:            [bs, F]
        # highlight_mask:       [bs, F]

        # [bs, F, d]
        embeddings = self.selector_embedder.forward(
            features=data.features, mask=highlight_mask
        )

        # [bs, F, d_2]
        encodings = self.encode_features(
            embeddings=embeddings, mask=highlight_mask, encoder=self.predictor_encoder
        )

        # [bs, F, d_3]
        pooled_encodings = self.pool_encodings(
            encodings=encodings, mask=highlight_mask, encoder=self.predictor_encoder
        )

        # [bs, C]
        predictor_logits = self.predictor.forward(encodings=pooled_encodings)

        return predictor_logits

    def forward(self, data: SPPInputData) -> SPPOutputData:
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

        return SPPOutputData(selectors_logits=selectors_logits,
                             predictors_logits=predictors_logits,
                             highlight_masks=highlight_masks)

    # TODO: hard to inherit for subclasses. Find a better way
    def compute_loss(
            self,
            input_data: SPPInputData,
            output_data: SPPOutputData
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total_loss = th.Tensor(0.0, device=input_data.y_true.device)
        losses = {}

        # [bs,]
        sample_weights = th.where(input_data.y_true == 1, self.class_weights[1], self.class_weights[0])

        # y_hat:            [bs, F, C]
        # highlight_mask:   [bs, F]
        for y_hat, highlight_mask in zip(th.unbind(output_data.predictors_logits, dim=1),
                                         th.unbind(output_data.highlight_masks, dim=1)):
            clf_loss = self.clf_loss(y_hat, input_data.y_true)
            sample_weights = sample_weights.to(clf_loss.device)
            clf_loss = (clf_loss * sample_weights).sum() / sample_weights.sum()

            total_loss += clf_loss
            losses['CE'] = losses.get('CE', 0) + clf_loss

        return total_loss, losses

    def training_step(self, batch: SPPInputData, batch_idx: int):
        output_data = self.forward(data=batch)

        batch_size = output_data.selectors_logits.shape[0]

        total_loss, losses = self.compute_loss(input_data=batch,
                                               output_data=output_data)

        self.log(
            name="train_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"train_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
            )

        # TODO: define aggregator for computing metrics!
        if self.train_classification_metrics is not None:
            y_hat = th.argmax(output_data.predictors_logits, dim=-1)
            self.train_classification_metrics.update(y_hat, batch.y_true)

        # TODO: SPPInputData and SPPOutputData could implement a method to simplify this
        # if self.store_predictions:
        #     self.predictions.append(
        #         [
        #             input_ids.detach().cpu().numpy(),
        #             y_hat.detach().cpu().numpy(),
        #             y_true.detach().cpu().numpy(),
        #         ]
        #     )

        return total_loss
