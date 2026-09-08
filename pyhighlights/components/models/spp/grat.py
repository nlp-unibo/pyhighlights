import abc
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, OutputData
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility.losses import Loss, build_losses, compute_losses


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
    """Guider-regularized rationalizer with staged optimization.

    A soft attention classifier over the full input is pretrained, then keeps
    training alongside the rationalizer: its attention supervises the
    selection and its predictions are matched in distribution, so the
    generator is guided instead of regularized ad hoc.

    ``losses`` scores the rationalizer over a namespace holding the guider
    fields (``selection_logits``, ``guide_target``, ``guider_class_logits``)
    next to the model ones; ``guider_losses`` scores the guider alone. The
    guide and JSD terms are annealed against each other by name.

    Hu and Yu, 2024, *Learning Robust Rationales for Model Explainability: A
    Guidance-Based Approach*, AAAI 2024, 18243-18251.
    Reference implementation: <https://github.com/shuaibo919/g-rat>.
    """

    def __init__(
        self,
        selector_backbones: RegistrationKey[SPPBackbone],
        selectors: RegistrationKey[SPPSelector],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        guider: RegistrationKey[GRATGuider],
        guider_losses: List[RegistrationKey[Loss]],
        pretrain_epochs: int = 10,
        guide_decay: float = 1e-4,
        guide_loss: str = "guide",
        jsd_loss: str = "jsd",
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("G-RAT requires a separate predictor backbone")
        if not math.isfinite(guide_decay) or guide_decay < 0:
            raise ValueError("guide_decay must be finite and non-negative")
        if pretrain_epochs < 0:
            raise ValueError("pretrain_epochs must be non-negative")
        super().__init__(
            selector_backbones=selector_backbones,
            selectors=selectors,
            predictor=predictor,
            predictor_backbone=predictor_backbone,
            **kwargs,
        )
        if len(self.selectors) != 1:
            raise ValueError("G-RAT requires exactly one selector")
        self.guider = Registry.from_key(guider, expected_type=GRATGuider)
        self.guider_losses = th.nn.ModuleList(build_losses(guider_losses))
        self.pretrain_epochs = pretrain_epochs
        self.guide_decay = guide_decay
        self.guide_loss = guide_loss
        self.jsd_loss = jsd_loss
        self.register_buffer("_model_steps", th.zeros((), dtype=th.long))
        self.automatic_optimization = False

    @property
    def guide_factor(self) -> float:
        completed_decay_steps = max(self._model_steps.item() - 1, 0)
        return max(1.0 - completed_decay_steps * self.guide_decay, 0.0)

    def guider_loss(
        self, input_data: InputData, output_data: GRATGuiderOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        return compute_losses(
            self.guider_losses,
            self.namespace(
                input_data, OutputData(class_logits=output_data.class_logits)
            ),
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
        values = self.head_namespace(
            input_data,
            output_data,
            selection_logits=output_data.highlight_logits[:, 0, :, 1],
            guide_target=self.guide_target(
                guider_output.attention.detach(), input_data.mask
            ),
            guider_class_logits=guider_output.class_logits.detach(),
        )
        factor = self.guide_factor
        return compute_losses(
            self.losses,
            values,
            scales={self.guide_loss: factor, self.jsd_loss: 1 - factor},
        )

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
        guider_total, guider_losses = self.guider_loss(batch, guider_output)
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
            losses={
                **{f"guider_{name}": value for name, value in guider_losses.items()},
                **model_losses,
            },
            batch_size=batch.y_true.shape[0],
        )
        self.update_metrics(split="train", input_data=batch, output_data=output)
        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output.as_numpy()})
        return total
