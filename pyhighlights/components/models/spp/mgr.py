from typing import Dict, Literal, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, Split
from pyhighlights.components.models.data import SPPOutput
from pyhighlights.components.models.spp.base import SPP, SPPBackbone


class MGR(SPP):
    """Multiple independent generators with one shared predictor.

    Generator ``i`` uses learning rate ``i * eta``; the predictor uses
    ``eta / n`` for ``n`` generators, following the original training policy.
    """

    def __init__(
        self,
        predictor_backbone: RegistrationKey[SPPBackbone] | None = None,
        inference_head: int = 0,
        loss_reduction: Literal["sum", "mean"] = "sum",
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("MGR requires a separate predictor backbone")
        super().__init__(predictor_backbone=predictor_backbone, **kwargs)
        if len(self.selectors) < 2:
            raise ValueError("MGR requires at least two generators")
        if not 0 <= inference_head < len(self.selectors):
            raise ValueError("inference_head is outside the generator range")
        if loss_reduction not in ("sum", "mean"):
            raise ValueError("loss_reduction must be 'sum' or 'mean'")
        self.inference_head = inference_head
        self.loss_reduction = loss_reduction

    def configure_optimizers(self):
        generators = [
            [*backbone.parameters(), *selector.parameters()]
            for backbone, selector in zip(self.selector_backbones, self.selectors)
        ]
        optimizer = Registry.from_key(
            self.optimizer,
            params=[
                {
                    "params": [
                        *self.predictor_backbone.parameters(),
                        *self.predictor.parameters(),
                    ]
                },
                *({"params": parameters} for parameters in generators),
            ],
        )
        scales = [1 / len(generators), *range(1, len(generators) + 1)]
        for group, scale in zip(optimizer.param_groups, scales):
            group["lr"] *= scale
        return optimizer

    def forward_one_head(
        self, data: InputData, selector_idx: int | None = None
    ) -> SPPOutput:
        selector_idx = self.inference_head if selector_idx is None else selector_idx
        if not 0 <= selector_idx < len(self.selectors):
            raise ValueError("selector_idx is outside the generator range")
        highlight_logits, highlight_mask = self.select(
            data=data,
            selector=self.selectors[selector_idx],
            backbone=self.selector_backbones[selector_idx],
        )
        class_logits = self.predict(data=data, highlight_mask=highlight_mask)
        return SPPOutput(
            class_logits=class_logits.unsqueeze(1),
            highlight_logits=highlight_logits.unsqueeze(1),
            highlight_mask=highlight_mask.unsqueeze(1),
        )

    def validation_forward(self, batch: InputData) -> SPPOutput:
        return self.forward_one_head(data=batch)

    def test_forward(self, batch: InputData) -> SPPOutput:
        return self.forward_one_head(data=batch)

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: SPPOutput
    ):
        if output_data.class_logits.shape[1] > 1:
            head = self.inference_head
            output_data = SPPOutput(
                class_logits=output_data.class_logits[:, head : head + 1],
                highlight_logits=output_data.highlight_logits[:, head : head + 1],
                highlight_mask=output_data.highlight_mask[:, head : head + 1],
            )
        super().update_metrics(split, input_data, output_data)

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total, losses = super().compute_loss(input_data, output_data)
        if self.loss_reduction == "mean":
            heads = output_data.class_logits.shape[1]
            return total / heads, {
                name: value / heads for name, value in losses.items()
            }
        return total, losses
