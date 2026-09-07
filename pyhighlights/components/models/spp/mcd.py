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
from pyhighlights.utility.losses import KLDiv, Loss


class MCD(SPP):
    """Rationalizer trained against selected-input and full-input predictions."""

    def __init__(
        self,
        selector_backbones: RegistrationKey[SPPBackbone],
        selectors: RegistrationKey[SPPSelector],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        classification_loss: RegistrationKey[Loss],
        rationale_losses: List[RegistrationKey[Loss]],
        discrepancy_coefficient: float = 1.0,
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("MCD requires a separate predictor backbone")
        if discrepancy_coefficient < 0:
            raise ValueError("discrepancy_coefficient must be non-negative")
        super().__init__(
            selector_backbones=selector_backbones,
            selectors=selectors,
            predictor=predictor,
            predictor_backbone=predictor_backbone,
            losses=[classification_loss, *rationale_losses],
            **kwargs,
        )
        if len(self.selectors) != 1:
            raise ValueError("MCD requires exactly one selector")
        self.discrepancy = KLDiv()
        self.discrepancy_coefficient = discrepancy_coefficient
        self.automatic_optimization = False

    @property
    def classification_loss(self) -> Loss:
        return self.losses[0]

    @property
    def rationale_losses(self):
        return self.losses[1:]

    def predict_full(self, data: InputData) -> th.Tensor:
        return self.predict(data=data, highlight_mask=data.mask)

    def _output(
        self,
        class_logits: th.Tensor,
        highlight_logits: th.Tensor,
        highlight_mask: th.Tensor,
    ) -> SPPOutput:
        return SPPOutput(
            class_logits=class_logits.unsqueeze(1),
            highlight_logits=highlight_logits.unsqueeze(1),
            highlight_mask=highlight_mask.unsqueeze(1),
        )

    def _rationale_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        head_output = next(output_data.unbind(dim=1))
        total = output_data.class_logits.new_zeros(())
        values = {}
        for loss in self.rationale_losses:
            if not loss.enabled:
                continue
            value = loss(input_data=input_data, output_data=head_output)
            total = total + value * loss.coefficient
            values[loss.name] = value
        return total, values

    def _classification_loss(
        self, input_data: InputData, class_logits: th.Tensor
    ) -> th.Tensor:
        return self.classification_loss(
            input_data=input_data,
            output_data=OutputData(class_logits=class_logits),
        )

    def classifier_phase_loss(
        self, input_data: InputData
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor], SPPOutput]:
        highlight_logits, highlight_mask = self.select(
            data=input_data,
            selector=self.selectors[0],
            backbone=self.selector_backbones[0],
        )
        selected_logits = self.predict(input_data, highlight_mask.detach())
        output = self._output(selected_logits, highlight_logits, highlight_mask)
        rationale_total, values = self._rationale_loss(input_data, output)

        selected = self._classification_loss(input_data, selected_logits)
        full = self._classification_loss(input_data, self.predict_full(input_data))
        coefficient = self.classification_loss.coefficient
        total = rationale_total + coefficient * (selected + full)
        return (
            total,
            {
                **values,
                "selected_classification": selected,
                "full_classification": full,
            },
            output,
        )

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
            highlight_logits, highlight_mask = self.select(
                data=input_data,
                selector=self.selectors[0],
                backbone=self.selector_backbones[0],
            )
            selected_logits = self.predict(input_data, highlight_mask)
            full_logits = self.predict_full(input_data)
            output = self._output(selected_logits, highlight_logits, highlight_mask)
            rationale_total, values = self._rationale_loss(input_data, output)
            discrepancy = self.discrepancy(selected_logits, full_logits)
            total = rationale_total + discrepancy * self.discrepancy_coefficient
            return total, {**values, "discrepancy": discrepancy}, output
        finally:
            for parameter, enabled in zip(predictor_parameters, requires_grad):
                parameter.requires_grad_(enabled)

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        if output_data.class_logits.shape[1] != 1:
            raise ValueError("MCD output must contain exactly one head")
        selected_logits = output_data.class_logits[:, 0]
        selected = self._classification_loss(input_data, selected_logits)
        rationale_total, values = self._rationale_loss(input_data, output_data)
        discrepancy = self.discrepancy(selected_logits, self.predict_full(input_data))
        total = (
            selected * self.classification_loss.coefficient
            + rationale_total
            + discrepancy * self.discrepancy_coefficient
        )
        return total, {
            **values,
            "selected_classification": selected,
            "discrepancy": discrepancy,
        }

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
        self.log_metrics(
            split="train",
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
            batch_size=batch.y_true.shape[0],
        )
        self.update_metrics(split="train", input_data=batch, output_data=output)
        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output.as_numpy()})
        return total
