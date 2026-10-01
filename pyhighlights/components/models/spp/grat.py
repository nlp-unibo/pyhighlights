import abc
import contextlib
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

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
from pyhighlights.utility import diagnostics
from pyhighlights.utility.losses import Loss, build_losses, compute_losses


@dataclass
class GRATGuiderOutput:
    """What a guider reads off the full input."""

    #: ``[B, T]``: attention over the encoder axis, summing to one per row.
    attention: th.Tensor
    #: ``[B, C]``: the guider's class logits.
    class_logits: th.Tensor


class GRATGuider(th.nn.Module, abc.ABC):
    """An attention classifier over the full input, which guides the selector."""

    @abc.abstractmethod
    def forward(self, data: InputData, mask: th.Tensor) -> GRATGuiderOutput:
        """Return normalized token attention [B, T] and logits [B, C].

        ``mask`` is what the guider's encoder attends over, on the encoder
        axis. G-RAT passes ``encoder_mask``, which is the only axis a subword
        backbone can encode, and folds the attention back onto the selection
        axis itself.
        """


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

    def forward(self, data: InputData, mask: th.Tensor) -> GRATGuiderOutput:
        valid = mask.bool()
        states = self.backbone.encode(data.features, mask)
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
        # The annealing scales losses by name, and a name matching no loss
        # scales nothing, so a misnamed term would train at full weight.
        names = [loss.name for loss in self.losses]
        for field, loss_name in (("guide_loss", guide_loss), ("jsd_loss", jsd_loss)):
            if loss_name not in names:
                raise ValueError(
                    f"{field} {loss_name!r} names no loss of this model, "
                    f"whose losses are {names}"
                )
        self.guider = Registry.from_key(guider, expected_type=GRATGuider)
        self.guider_losses = th.nn.ModuleList(build_losses(guider_losses))
        self.pretrain_epochs = pretrain_epochs
        self.guide_decay = guide_decay
        self.guide_loss = guide_loss
        self.jsd_loss = jsd_loss
        self.register_buffer("_model_steps", th.zeros((), dtype=th.long))
        self.automatic_optimization = False

    def loss_names(self) -> List[str]:
        # The guider's terms are logged under a prefix beside the model's.
        guider = [f"guider_{loss.name}" for loss in self.guider_losses]
        return [*super().loss_names(), *guider]

    @property
    def warmup_epochs(self) -> int:
        """The guider's pretraining, which the rationalizer sits out.

        ``training_step`` gates the model's optimizer on
        ``current_epoch >= pretrain_epochs``, so until then the monitored
        quantities describe a model that has taken no step, and a monitor
        counting from epoch zero can stop a run inside this window.
        """
        return int(self.pretrain_epochs)

    @property
    def guide_factor(self) -> th.Tensor:
        """The guide term's weight, ``max(1 - max(t - 1, 0) * guide_decay, 0)``.

        ``t`` counts the rationalizer's steps. The reference implementation's
        ``FactorAnnealer`` applies its decay before counting the step, so the
        first two steps both train at a weight of one. A tensor rather than a
        float, so reading it does not synchronise with the device.
        """
        completed = (self._model_steps - 1).clamp_min(0)
        return (1.0 - completed * self.guide_decay).clamp_min(0.0)

    def guider_loss(
        self, input_data: InputData, output_data: GRATGuiderOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        return compute_losses(
            self.guider_losses,
            self.namespace(
                input_data, OutputData(class_logits=output_data.class_logits)
            ),
        )

    def to_selection_axis(self, attention: th.Tensor, data: InputData) -> th.Tensor:
        """The guider's attention on the axis the selection is scored on.

        The guider attends over subtokens because that is what its encoder
        reads. The selection it guides is over words, so each word takes the
        attention its subtokens hold between them, summed, since attention is
        a distribution. A model selecting over subtokens, or a vocabulary
        tokenizer, needs no folding and gets none.
        """
        return self.to_words(attention.unsqueeze(-1), data, reduce="sum").squeeze(-1)

    def guide_target(self, attention: th.Tensor, mask: th.Tensor) -> th.Tensor:
        """The per-word target the guide term trains the selection towards.

        ``min(a_i / (mean(a) + 1 / (1 + n)), 1)`` over the ``n`` valid words,
        zero elsewhere. The mean is taken over valid words. The reference
        implementation takes it over the padded width, which makes a target
        depend on the widest document in its batch.
        """
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
            selection_logits=self.reported(output_data).highlight_logits[..., 1],
            guide_target=self.guide_target(
                self.to_selection_axis(guider_output.attention.detach(), input_data),
                self.selection_valid(input_data),
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
            guider_output = self.guider(input_data, self.encoder_mask(input_data))
        return self.model_loss(input_data, output_data, guider_output)

    def guider_encoder_ids(self) -> Set[int]:
        """The guider's own encoder, which is pretrained when the model's is.

        ``encoder_ids`` covers the backbones a rationalizer reads with. The
        guider holds a third, and a rate meant for pretrained encoders that
        skipped it would fine-tune one of the three at the selector's rate.
        """
        return {id(parameter) for parameter in self.guider.backbone.parameters()}

    def encoder_ids(self) -> Set[int]:
        return super().encoder_ids() | self.guider_encoder_ids()

    def configure_optimizers(self):
        model_parameters = [
            *self.generator_parameters(),
            *self.predictor_parameters(),
        ]
        # Two optimizers, as the reference implementation has: the guider is
        # stepped before the rationalizer and on its own loss. Both are built
        # through `build_optimizer`, so `encoder_lr` reaches the guider's
        # encoder as well as the model's.
        return [
            self.build_optimizer([(self.guider.parameters(), 1.0)]),
            self.build_optimizer([(model_parameters, 1.0)]),
        ]

    def backward_and_average(
        self, loss: th.Tensor, parameters: Iterable[th.nn.Parameter]
    ) -> None:
        """Backpropagate ``loss`` and average the gradients of ``parameters``.

        Each of G-RAT's two backward passes reaches one half of the model, so
        a data-parallel wrapper, which expects every registered parameter in
        every pass, would raise. The wrapper's synchronisation is blocked for
        the pass, and the half that pass trains is averaged across processes
        here instead. On one process the average is the identity.
        """
        strategy = self.trainer.strategy
        blocked = getattr(strategy, "block_backward_sync", contextlib.nullcontext)
        with blocked():
            self.manual_backward(loss)
        for parameter in parameters:
            if parameter.grad is not None:
                parameter.grad = strategy.reduce(parameter.grad, reduce_op="mean")

    def training_step(self, batch: InputData, batch_idx: int):
        # As in `PhasedSPP`: a manual step marks its own split.
        diagnostics.record("step", split="train", batch=batch_idx)
        guider_optimizer, model_optimizer = self.optimizers()
        guider_optimizer.zero_grad()
        guider_output = self.guider(batch, self.encoder_mask(batch))
        guider_total, guider_losses = self.guider_loss(batch, guider_output)
        self.backward_and_average(guider_total, self.guider.parameters())
        guider_optimizer.step()
        guider_optimizer.zero_grad()

        output = self(batch)
        was_training = self.guider.training
        self.guider.eval()
        with th.no_grad():
            guider_output = self.guider(batch, self.encoder_mask(batch))
        self.guider.train(was_training)
        model_total, model_losses = self.model_loss(batch, output, guider_output)

        if self.current_epoch >= self.pretrain_epochs:
            model_optimizer.zero_grad()
            self.backward_and_average(
                model_total,
                [*self.generator_parameters(), *self.predictor_parameters()],
            )
            model_optimizer.step()
            model_optimizer.zero_grad()
            self._model_steps.add_(1)

        total = model_total.detach()
        self.record(
            split="train",
            batch=batch,
            output_data=output,
            total_loss=total,
            losses={
                **{f"guider_{name}": value for name, value in guider_losses.items()},
                **model_losses,
            },
        )
        return total
