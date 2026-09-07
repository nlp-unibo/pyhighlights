import abc
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, OutputData
from pyhighlights.components.models.data import SPPOutput
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.utility.losses import JSDiv, Loss


@dataclass
class GRATGuiderOutput:
    attention: th.Tensor
    class_logits: th.Tensor


class GRATGuider(th.nn.Module, abc.ABC):
    @abc.abstractmethod
    def forward(self, data: InputData) -> GRATGuiderOutput:
        """Return normalized token attention [B, T] and logits [B, C]."""


class AttentionGuider(GRATGuider):
    """Backbone-independent attention classifier used to guide G-RAT."""

    def __init__(
        self,
        backbone: RegistrationKey[SPPBackbone],
        predictor: RegistrationKey[SPPPredictor],
        noise_sigma: float = 1.0,
    ):
        super().__init__()
        if not math.isfinite(noise_sigma) or noise_sigma < 0:
            raise ValueError("noise_sigma must be finite and non-negative")
        self.backbone = Registry.from_key(backbone, expected_type=SPPBackbone)
        self.attention = th.nn.Linear(self.backbone.output_size, 1)
        self.predictor = Registry.from_key(
            predictor,
            expected_type=SPPPredictor,
            input_size=self.backbone.output_size,
        )
        self.noise_sigma = noise_sigma

    def forward(self, data: InputData) -> GRATGuiderOutput:
        valid = data.mask.bool()
        states = self.backbone.encode(data.features, data.mask)
        scores = self.attention(states).squeeze(-1)
        if self.training and self.noise_sigma:
            scores = scores + th.randn_like(scores).abs() * self.noise_sigma
        scores = scores.masked_fill(~valid, -th.inf)
        scores = th.where(valid.any(dim=1, keepdim=True), scores, th.zeros_like(scores))
        attention = th.softmax(scores, dim=1) * valid.to(scores.dtype)
        pooled = (states * attention.unsqueeze(-1)).sum(dim=1)
        return GRATGuiderOutput(
            attention=attention,
            class_logits=self.predictor(pooled),
        )


class GRAT(SPP):
    """Guider-regularized rationalizer with staged optimization."""

    def __init__(
        self,
        selector_backbones: RegistrationKey[SPPBackbone],
        selectors: RegistrationKey[SPPSelector],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        guider: RegistrationKey[GRATGuider],
        classification_loss: RegistrationKey[Loss],
        rationale_losses: List[RegistrationKey[Loss]],
        pretrain_epochs: int = 10,
        guide_coefficient: float = 1.0,
        jsd_coefficient: float = 1.0,
        guide_decay: float = 1e-4,
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("G-RAT requires a separate predictor backbone")
        values = (guide_coefficient, jsd_coefficient, guide_decay)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("G-RAT coefficients must be finite and non-negative")
        if pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative")
        super().__init__(
            selector_backbones=selector_backbones,
            selectors=selectors,
            predictor=predictor,
            predictor_backbone=predictor_backbone,
            losses=[classification_loss, *rationale_losses],
            **kwargs,
        )
        if len(self.selectors) != 1:
            raise ValueError("G-RAT requires exactly one selector")
        self.guider = Registry.from_key(guider, expected_type=GRATGuider)
        self.pretrain_epochs = pretrain_epochs
        self.guide_coefficient = guide_coefficient
        self.jsd_coefficient = jsd_coefficient
        self.guide_decay = guide_decay
        self.jsd = JSDiv()
        self.register_buffer("_model_steps", th.zeros((), dtype=th.long))
        self.automatic_optimization = False

    @property
    def classification_loss(self) -> Loss:
        return self.losses[0]

    @property
    def guide_factor(self) -> float:
        completed_decay_steps = max(self._model_steps.item() - 1, 0)
        return max(1.0 - completed_decay_steps * self.guide_decay, 0.0)

    def guider_loss(
        self, input_data: InputData, output_data: GRATGuiderOutput
    ) -> th.Tensor:
        return self.classification_loss(
            input_data=input_data,
            output_data=OutputData(class_logits=output_data.class_logits),
        )

    def guide_target(self, attention: th.Tensor, mask: th.Tensor) -> th.Tensor:
        valid = mask.bool()
        count = valid.sum(dim=1, keepdim=True).clamp_min(1)
        mean = (attention * valid).sum(dim=1, keepdim=True) / count
        scaling = mean + 1 / (1 + count)
        return (attention / scaling.clamp_min(1e-8)).clamp_max(1) * valid

    def model_loss(
        self,
        input_data: InputData,
        output_data: SPPOutput,
        guider_output: GRATGuiderOutput,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total, losses = super().compute_loss(input_data, output_data)
        valid = input_data.mask.bool()
        guide_target = self.guide_target(
            guider_output.attention.detach(), input_data.mask
        )
        selection_logits = output_data.highlight_logits[:, 0, :, 1]
        if valid.any():
            guide = th.nn.functional.binary_cross_entropy_with_logits(
                selection_logits[valid], guide_target[valid]
            )
        else:
            guide = selection_logits.sum() * 0
        jsd = self.jsd(
            output_data.class_logits[:, 0], guider_output.class_logits.detach()
        )
        factor = self.guide_factor
        total = (
            total
            + guide * self.guide_coefficient * factor
            + jsd * self.jsd_coefficient * (1 - factor)
        )
        return total, {**losses, "guide": guide, "jsd": jsd}

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        with th.no_grad():
            guider_output = self.guider(input_data)
        return self.model_loss(input_data, output_data, guider_output)

    def configure_optimizers(self):
        model_parameters = [
            *self.selector_backbones.parameters(),
            *self.selectors.parameters(),
            *self.predictor_backbone.parameters(),
            *self.predictor.parameters(),
        ]
        return [
            Registry.from_key(self.optimizer, params=self.guider.parameters()),
            Registry.from_key(self.optimizer, params=model_parameters),
        ]

    def training_step(self, batch: InputData, batch_idx: int):
        guider_optimizer, model_optimizer = self.optimizers()
        guider_optimizer.zero_grad()
        guider_output = self.guider(batch)
        guider_total = self.guider_loss(batch, guider_output)
        self.manual_backward(guider_total)
        guider_optimizer.step()
        guider_optimizer.zero_grad()

        output = self(batch)
        was_training = self.guider.training
        self.guider.eval()
        with th.no_grad():
            guider_output = self.guider(batch)
        self.guider.train(was_training)
        model_total, model_losses = self.model_loss(batch, output, guider_output)

        if self.current_epoch >= self.pretrain_epochs:
            model_optimizer.zero_grad()
            self.manual_backward(model_total)
            model_optimizer.step()
            model_optimizer.zero_grad()
            self._model_steps.add_(1)

        total = model_total.detach()
        self.log_metrics(
            split="train",
            total_loss=total,
            losses={"guider_classification": guider_total, **model_losses},
            batch_size=batch.y_true.shape[0],
        )
        self.update_metrics(split="train", input_data=batch, output_data=output)
        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output.as_numpy()})
        return total
