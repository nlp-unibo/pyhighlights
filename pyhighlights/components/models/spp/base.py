from __future__ import annotations

import abc
import math
from typing import Dict, List, Tuple, Union

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, Model, OutputData, Split
from pyhighlights.components.models.data import SPPOutput


class SPPBackbone(th.nn.Module, abc.ABC):
    """Backend-specific token encoder and pooler."""

    @property
    @abc.abstractmethod
    def output_size(self) -> int:
        """Token and pooled state width."""

    @abc.abstractmethod
    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        """Return token states shaped [B, T, D]."""

    @abc.abstractmethod
    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """Return sequence states shaped [B, D]."""


class SPPSelector(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, states: th.Tensor) -> th.Tensor:
        """Return selection logits shaped [B, T, 2]."""


class SPPPredictor(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, states: th.Tensor) -> th.Tensor:
        """Return class logits shaped [B, C]."""


class SPPAggregator(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, output_data: SPPOutput) -> OutputData: ...


class SPPFirstAggregator(SPPAggregator):
    def forward(self, output_data: SPPOutput) -> SPPOutput:
        return next(output_data.unbind(dim=1))


class SPP(Model):
    def __init__(
        self,
        selector_backbones: Union[
            RegistrationKey[SPPBackbone], List[RegistrationKey[SPPBackbone]]
        ],
        selectors: Union[
            RegistrationKey[SPPSelector], List[RegistrationKey[SPPSelector]]
        ],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None = None,
        aggregator: RegistrationKey[SPPAggregator] | None = None,
        temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        backbone_keys = (
            [selector_backbones]
            if isinstance(selector_backbones, RegistrationKey)
            else selector_backbones
        )
        selector_keys = (
            [selectors] if isinstance(selectors, RegistrationKey) else selectors
        )
        if not backbone_keys or len(backbone_keys) != len(selector_keys):
            raise ValueError("SPP requires one selector backbone per selector")

        self.selector_backbones = th.nn.ModuleList(
            Registry.from_keys(backbone_keys, expected_type=SPPBackbone)
        )
        self.selectors = th.nn.ModuleList(
            Registry.from_key(
                key,
                expected_type=SPPSelector,
                input_size=backbone.output_size,
            )
            for key, backbone in zip(selector_keys, self.selector_backbones)
        )

        self.predictor_backbone = (
            Registry.from_key(predictor_backbone, expected_type=SPPBackbone)
            if predictor_backbone is not None
            else self.selector_backbones[0]
        )
        self.predictor = Registry.from_key(
            predictor,
            expected_type=SPPPredictor,
            input_size=self.predictor_backbone.output_size,
        )
        self.aggregator = (
            Registry.from_key(aggregator, expected_type=SPPAggregator)
            if aggregator is not None
            else SPPFirstAggregator()
        )
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and greater than zero")
        self.temperature = temperature

    @property
    def selector_backbone(self) -> SPPBackbone:
        return self.selector_backbones[0]

    def select_activation(self, highlight_logits: th.Tensor) -> th.Tensor:
        if self.training:
            return th.nn.functional.gumbel_softmax(
                highlight_logits, tau=self.temperature, hard=True, dim=-1
            )[..., 1]
        return highlight_logits.argmax(dim=-1).to(highlight_logits.dtype)

    def select(
        self,
        data: InputData,
        selector: SPPSelector,
        backbone: SPPBackbone,
    ) -> Tuple[th.Tensor, th.Tensor]:
        states = backbone.encode(data.features, data.mask)
        highlight_logits = selector(states)
        highlight_mask = self.select_activation(highlight_logits)
        valid = data.mask.bool()
        highlight_mask = highlight_mask * valid.to(highlight_mask.dtype)

        needs_fallback = valid.any(dim=1) & ~highlight_mask.bool().any(dim=1)
        if needs_fallback.any():
            scores = th.softmax(highlight_logits / self.temperature, dim=-1)[..., 1]
            scores = scores * valid.to(scores.dtype)
            fallback_hard = th.nn.functional.one_hot(
                scores.masked_fill(~valid, -th.inf).argmax(dim=1),
                num_classes=scores.shape[1],
            ).to(scores.dtype)
            fallback = fallback_hard + scores - scores.detach()
            highlight_mask = th.where(
                needs_fallback.unsqueeze(1), fallback, highlight_mask
            )

        return highlight_logits, highlight_mask

    def predict(self, data: InputData, highlight_mask: th.Tensor) -> th.Tensor:
        prediction_mask = data.mask.to(highlight_mask.dtype) * highlight_mask
        states = self.predictor_backbone.encode(
            data.features, data.mask, selection_mask=prediction_mask
        )
        pooled = self.predictor_backbone.pool(states, prediction_mask)
        return self.predictor(pooled)

    def forward(self, data: InputData) -> SPPOutput:
        highlight_logits = []
        highlight_masks = []
        class_logits = []

        for backbone, selector in zip(self.selector_backbones, self.selectors):
            head_logits, head_mask = self.select(
                data=data, selector=selector, backbone=backbone
            )
            highlight_logits.append(head_logits)
            highlight_masks.append(head_mask)
            class_logits.append(self.predict(data=data, highlight_mask=head_mask))

        return SPPOutput(
            class_logits=th.stack(class_logits, dim=1),
            highlight_logits=th.stack(highlight_logits, dim=1),
            highlight_mask=th.stack(highlight_masks, dim=1),
        )

    def head_namespace(
        self, input_data: InputData, output_data: SPPOutput, **extra: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """Namespace of the first head, with the head dimension dropped."""
        return self.namespace(input_data, next(output_data.unbind(dim=1)), **extra)

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: SPPOutput
    ):
        super().update_metrics(
            split=split,
            input_data=input_data,
            output_data=self.aggregator(output_data),
        )

    def compute_loss(
        self,
        input_data: InputData,
        output_data: SPPOutput,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total_loss = output_data.class_logits.new_zeros(())
        losses: Dict[str, th.Tensor] = {}

        for head_output in output_data.unbind(dim=1):
            head_loss, head_losses = super().compute_loss(input_data, head_output)
            total_loss = total_loss + head_loss
            for name, value in head_losses.items():
                losses[name] = losses.get(name, value.new_zeros(())) + value

        return total_loss, losses
