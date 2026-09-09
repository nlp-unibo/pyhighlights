from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility.losses import Loss, build_losses, compute_losses


class MCD(SPP):
    """Rationalizer trained against selected-input and full-input predictions.

    A predictor reading the full input guides the generator: a highlight that
    d-separates the label from the rest of the input makes the selected-input
    and full-input predictions agree.

    Liu, Wang, Wang, Li, Deng, Zhang and Qiu, 2023, *D-Separation for Causal
    Self-Explanation*, NeurIPS 2023.
    Reference implementation:
    <https://github.com/jugechengzi/Rationalization-MCD>.

    Losses are grouped per training phase: ``rationale_losses`` apply to both
    phases, ``predictor_losses`` only to the predictor phase and
    ``generator_losses`` only to the generator phase. Evaluation reports every
    group. Each namespace exposes ``full_class_logits`` next to the
    selected-input fields.
    """

    def __init__(
        self,
        selector_backbones: RegistrationKey[SPPBackbone],
        selectors: RegistrationKey[SPPSelector],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        rationale_losses: List[RegistrationKey[Loss]],
        predictor_losses: List[RegistrationKey[Loss]],
        generator_losses: List[RegistrationKey[Loss]],
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("MCD requires a separate predictor backbone")
        # MCD groups its losses per training phase and replaces the flat list
        # SPP builds, so a supervision loss appended to that list would be
        # dropped before the first batch. Refused rather than silently ignored:
        # a run that reports itself as the supervised ceiling and trains
        # unsupervised is a number nobody can read as wrong.
        if kwargs.get("supervise_highlights"):
            raise ValueError(
                "MCD scores its losses per training phase, so highlight "
                "supervision has to name the phase it belongs to; put the "
                "highlight loss in rationale_losses instead"
            )
        super().__init__(
            selector_backbones=selector_backbones,
            selectors=selectors,
            predictor=predictor,
            predictor_backbone=predictor_backbone,
            losses=[],
            **kwargs,
        )
        if len(self.selectors) != 1:
            raise ValueError("MCD requires exactly one selector")

        self.rationale_losses = th.nn.ModuleList(build_losses(rationale_losses))
        self.predictor_losses = th.nn.ModuleList(build_losses(predictor_losses))
        self.generator_losses = th.nn.ModuleList(build_losses(generator_losses))
        self.losses = th.nn.ModuleList(
            [*self.rationale_losses, *self.predictor_losses, *self.generator_losses]
        )
        self.automatic_optimization = False

    def phase_forward(
        self, input_data: InputData, detach_selection: bool
    ) -> Tuple[SPPOutput, Dict[str, th.Tensor]]:
        highlight_logits, highlight_mask = self.select(
            data=input_data,
            selector=self.selectors[0],
            backbone=self.selector_backbones[0],
        )
        selection = highlight_mask.detach() if detach_selection else highlight_mask
        output = SPPOutput(
            class_logits=self.predict(input_data, selection).unsqueeze(1),
            highlight_logits=highlight_logits.unsqueeze(1),
            highlight_mask=highlight_mask.unsqueeze(1),
        )
        values = self.head_namespace(
            input_data, output, full_class_logits=self.predict_full(input_data)
        )
        return output, values

    def classifier_phase_loss(
        self, input_data: InputData
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor], SPPOutput]:
        output, values = self.phase_forward(input_data, detach_selection=True)
        total, losses = compute_losses(
            [*self.rationale_losses, *self.predictor_losses], values
        )
        return total, losses, output

    def generator_phase_loss(
        self, input_data: InputData
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor], SPPOutput]:
        predictor_parameters = [
            *self.predictor_backbone.parameters(),
            *self.predictor.parameters(),
        ]
        requires_grad = [parameter.requires_grad for parameter in predictor_parameters]
        for parameter in predictor_parameters:
            parameter.requires_grad_(False)
        try:
            output, values = self.phase_forward(input_data, detach_selection=False)
            total, losses = compute_losses(
                [*self.rationale_losses, *self.generator_losses], values
            )
            return total, losses, output
        finally:
            for parameter, enabled in zip(predictor_parameters, requires_grad):
                parameter.requires_grad_(enabled)

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        if output_data.class_logits.shape[1] != 1:
            raise ValueError("MCD output must contain exactly one head")
        values = self.head_namespace(
            input_data, output_data, full_class_logits=self.predict_full(input_data)
        )
        return compute_losses(self.losses, values)

    def configure_optimizers(self):
        generator_parameters = [
            *self.selector_backbones.parameters(),
            *self.selectors.parameters(),
        ]
        predictor_parameters = [
            *self.predictor_backbone.parameters(),
            *self.predictor.parameters(),
        ]
        return [
            Registry.from_key(self.optimizer, params=generator_parameters),
            Registry.from_key(self.optimizer, params=predictor_parameters),
        ]

    def training_step(self, batch: InputData, batch_idx: int):
        generator_optimizer, predictor_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

        classifier_total, classifier_losses, output = self.classifier_phase_loss(batch)
        self.manual_backward(classifier_total)
        generator_optimizer.step()
        predictor_optimizer.step()
        generator_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

        generator_total, generator_losses, _ = self.generator_phase_loss(batch)
        self.manual_backward(generator_total)
        generator_optimizer.step()
        generator_optimizer.zero_grad()

        total = classifier_total.detach() + generator_total.detach()
        self.record(
            split="train",
            batch=batch,
            output_data=output,
            total_loss=total,
            losses={
                **{
                    f"classifier_{name}": value
                    for name, value in classifier_losses.items()
                },
                **{
                    f"generator_{name}": value
                    for name, value in generator_losses.items()
                },
            },
        )
        return total
