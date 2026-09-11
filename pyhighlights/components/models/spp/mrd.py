from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility.losses import Loss, build_losses, compute_losses


class MRD(SPP):
    """Rationalizer trained by what is left once the highlight is removed.

    Maximum mutual information asks the highlight to predict the label, which
    a spurious feature correlated with the label answers just as well. MRD
    asks the opposite question: remove the highlight, and what remains should
    stop looking like the whole input. Removing plain noise or a spurious
    feature leaves the conditional distribution of the rest unchanged, so only
    the causal features move it -- which makes a corpus full of spurious
    features behave like a clean one rather than needing a penalty per
    spurious feature.

    Two consequences that make this model read oddly beside the others. The
    predictor never trains on the highlight: it is trained on the
    **complement** and on the full input, and the highlight pass exists only
    so the metrics have something to score. And the generator maximizes a
    divergence rather than minimizing one, which is a loss with a negative
    coefficient.

    Liu, Deng, Niu, Wang, Wang, Zhang and Li, 2024, *Is the MMI Criterion
    Necessary for Interpretability? Degenerating Non-causal Features to Plain
    Noise for Self-Rationalization*, NeurIPS 2024.
    Paper: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/d53d51e88d92d3723755f6d425bc513b-Abstract-Conference.html>.
    Reference implementation:
    <https://github.com/jugechengzi/Rationalization-MRD>.

    Losses are grouped per training phase, as in MCD: ``rationale_losses``
    apply to both phases, ``predictor_losses`` only to the predictor phase and
    ``generator_losses`` only to the generator phase. Every namespace carries
    ``complement_class_logits`` and ``full_class_logits`` beside the
    highlight's own fields.
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
            raise ValueError("MRD requires a separate predictor backbone")
        # As in MCD: the flat loss list is replaced by three phase-scoped
        # ones, so a supervision loss appended to it would be dropped before
        # the first batch rather than refused.
        if kwargs.get("supervise_highlights"):
            raise ValueError(
                "MRD scores its losses per training phase, so highlight "
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
            raise ValueError("MRD requires exactly one selector")

        self.rationale_losses = th.nn.ModuleList(build_losses(rationale_losses))
        self.predictor_losses = th.nn.ModuleList(build_losses(predictor_losses))
        self.generator_losses = th.nn.ModuleList(build_losses(generator_losses))
        self.losses = th.nn.ModuleList(
            [*self.rationale_losses, *self.predictor_losses, *self.generator_losses]
        )
        self.automatic_optimization = False

    def extra_logits(
        self, input_data: InputData, highlight_mask: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """The two passes MRD trains on: the complement, and the full input."""
        return {
            "complement_class_logits": self.predict_complement(
                input_data, highlight_mask
            ),
            "full_class_logits": self.predict_full(input_data),
        }

    def phase_forward(
        self, input_data: InputData, detach_selection: bool
    ) -> Tuple[SPPOutput, Dict[str, th.Tensor]]:
        highlight_logits, highlight_mask = self.select(
            data=input_data,
            selector=self.selectors[0],
            backbone=self.selector_backbones[0],
        )
        selection = highlight_mask.detach() if detach_selection else highlight_mask
        # Outside the graph: nothing trains on the highlight pass, and the
        # reference implementation keeps it for the same reason -- a number to
        # score the model by, not a term.
        with th.no_grad():
            class_logits = self.predict(input_data, highlight_mask.detach())
        output = SPPOutput(
            class_logits=class_logits.unsqueeze(1),
            highlight_logits=highlight_logits.unsqueeze(1),
            highlight_mask=highlight_mask.unsqueeze(1),
        )
        values = self.head_namespace(
            input_data, output, **self.extra_logits(input_data, selection)
        )
        return output, values

    def predictor_phase_loss(
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
            raise ValueError("MRD output must contain exactly one head")
        head = self.aggregator(output_data)
        values = self.head_namespace(
            input_data,
            output_data,
            **self.extra_logits(input_data, head.highlight_mask),
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
        # One optimizer per phase, as the phases alternate and the generator
        # step runs with the predictor frozen. Through `build_optimizer`, so
        # `encoder_lr` reaches the encoder inside each.
        return [
            self.build_optimizer([(generator_parameters, 1.0)]),
            self.build_optimizer([(predictor_parameters, 1.0)]),
        ]

    def training_step(self, batch: InputData, batch_idx: int):
        generator_optimizer, predictor_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

        # The predictor learns to read the complement and the full input; the
        # selection reaching it is detached, so nothing of that reaches the
        # generator except the sparsity and contiguity terms.
        predictor_total, predictor_losses, output = self.predictor_phase_loss(batch)
        self.manual_backward(predictor_total)
        predictor_optimizer.step()
        generator_optimizer.step()
        predictor_optimizer.zero_grad()
        generator_optimizer.zero_grad()

        generator_total, generator_losses, _ = self.generator_phase_loss(batch)
        self.manual_backward(generator_total)
        generator_optimizer.step()
        generator_optimizer.zero_grad()

        total = predictor_total.detach() + generator_total.detach()
        self.record(
            split="train",
            batch=batch,
            output_data=output,
            total_loss=total,
            losses={
                **{
                    f"predictor_{name}": value
                    for name, value in predictor_losses.items()
                },
                **{
                    f"generator_{name}": value
                    for name, value in generator_losses.items()
                },
            },
        )
        return total
