import abc
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
from pyhighlights.utility import diagnostics
from pyhighlights.utility.losses import Loss, build_losses, compute_losses


class PhasedSPP(SPP):
    """A rationalizer whose criteria belong to one of two training phases.

    Two architectures are built this way. The predictor phase trains the
    predictor on a selection it is handed and may not move; the generator
    phase trains the generator with the predictor frozen. What differs between
    them is which criteria go in which list and what the predictor is asked to
    read, and that is what the subclasses say.

    ``shared_losses`` are scored in both phases, ``predictor_losses`` in the
    predictor phase and ``generator_losses`` in the generator phase.
    Evaluation reports every group, since a validation number is about the
    model rather than about a phase of its training.

    Two properties of the loop read oddly until they are checked against the
    implementations it reproduces, so both are stated here.

    The shared criteria bind to the selection the generator produced rather
    than to the detached copy the predictor reads, so they reach the generator
    in the predictor phase as well as in its own, and the generator's
    optimizer is stepped in both. It therefore takes two steps per batch on
    the shared criteria and one on the phase-specific term. Both reference
    implementations do the same: ``train_util.train_decouple_causal2`` of
    <https://github.com/jugechengzi/Rationalization-MCD> adds the sparsity and
    continuity terms to its classification loss and steps ``opt_gen`` beside
    ``opt_pred``, and ``train_util.train_adv_causal`` of
    <https://github.com/jugechengzi/Rationalization-MRD> does the same under
    ``--gen_sparse``, which defaults to 1 and is what its README runs. A study
    that wants the other arrangement registers ``shared_losses`` empty, which
    is what turning that flag off amounts to.

    And each phase draws its own selection: ``phase_forward`` selects once per
    phase, so the two phases of a batch optimize different masks of it. That
    is the references again, which call ``get_rationale`` in each phase over a
    ``gumbel_softmax`` with ``hard=True``.
    """

    #: What this model's paper calls the phase that trains the predictor. It
    #: prefixes that phase's terms in the training log, so a model keeps the
    #: names its own results were written under.
    predictor_phase: str = "predictor"

    def __init__(
        self,
        selector_backbones: RegistrationKey[SPPBackbone],
        selectors: RegistrationKey[SPPSelector],
        predictor: RegistrationKey[SPPPredictor],
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        shared_losses: List[RegistrationKey[Loss]],
        predictor_losses: List[RegistrationKey[Loss]],
        generator_losses: List[RegistrationKey[Loss]],
        **kwargs,
    ):
        name = type(self).__name__
        if predictor_backbone is None:
            raise ValueError(f"{name} requires a separate predictor backbone")
        # A phased model groups its losses per training phase and replaces the
        # flat list SPP builds, so a supervision loss appended to that list
        # would be dropped before the first batch. Refused rather than
        # silently ignored: a run that reports itself as the supervised
        # ceiling and trains unsupervised is a number nobody can read as wrong.
        if kwargs.get("supervise_highlights"):
            raise ValueError(
                f"{name} scores its losses per training phase, so highlight "
                "supervision has to name the phase it belongs to; put the "
                "highlight loss in shared_losses instead"
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
            raise ValueError(f"{name} requires exactly one selector")

        self.shared_losses = th.nn.ModuleList(build_losses(shared_losses))
        self.predictor_losses = th.nn.ModuleList(build_losses(predictor_losses))
        self.generator_losses = th.nn.ModuleList(build_losses(generator_losses))
        self.losses = th.nn.ModuleList(
            [*self.shared_losses, *self.predictor_losses, *self.generator_losses]
        )
        self.automatic_optimization = False

    @abc.abstractmethod
    def phase_class_logits(
        self, input_data: InputData, highlight_mask: th.Tensor, selection: th.Tensor
    ) -> th.Tensor:
        """What the predictor makes of the highlight, however the model asks.

        ``highlight_mask`` is what the generator produced and ``selection`` is
        the copy this phase hands the predictor, detached in the predictor
        phase. A model that trains on the highlight reads the second; one that
        trains on the complement reads neither and takes this pass outside the
        graph.
        """

    @abc.abstractmethod
    def extra_logits(
        self, input_data: InputData, selection: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """The predictor passes this model's criteria bind to, beside the
        highlight's own fields."""

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
            class_logits=self.phase_class_logits(
                input_data, highlight_mask, selection
            ).unsqueeze(1),
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
        # Which phase the stages below belong to. A batch passes through them
        # twice here, and the two passes are different selections scored by
        # different criteria.
        diagnostics.record("phase", name=self.predictor_phase)
        output, values = self.phase_forward(input_data, detach_selection=True)
        total, losses = compute_losses(
            [*self.shared_losses, *self.predictor_losses], values
        )
        return total, losses, output

    def generator_phase_loss(
        self, input_data: InputData
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor], SPPOutput]:
        predictor_parameters = self.predictor_parameters()
        diagnostics.record("phase", name="generator")
        requires_grad = [parameter.requires_grad for parameter in predictor_parameters]
        for parameter in predictor_parameters:
            parameter.requires_grad_(False)
        try:
            output, values = self.phase_forward(input_data, detach_selection=False)
            total, losses = compute_losses(
                [*self.shared_losses, *self.generator_losses], values
            )
            return total, losses, output
        finally:
            for parameter, enabled in zip(predictor_parameters, requires_grad):
                parameter.requires_grad_(enabled)

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        name = type(self).__name__
        if output_data.class_logits.shape[1] != 1:
            raise ValueError(f"{name} output must contain exactly one head")
        head = self.aggregator(output_data)
        values = self.head_namespace(
            input_data,
            output_data,
            **self.extra_logits(input_data, head.highlight_mask),
        )
        return compute_losses(self.losses, values)

    def configure_optimizers(self):
        # One optimizer per phase, as the phases alternate and the generator
        # step runs with the predictor frozen. Through `build_optimizer`, so
        # `encoder_lr` reaches the encoder inside each.
        return [
            self.build_optimizer([(self.generator_parameters(), 1.0)]),
            self.build_optimizer([(self.predictor_parameters(), 1.0)]),
        ]

    def training_step(self, batch: InputData, batch_idx: int):
        # This model drives its own optimizers, so it never reaches
        # `Model._step` and has to say which split its stages are serving.
        diagnostics.record("step", split="train", batch=batch_idx)
        generator_optimizer, predictor_optimizer = self.optimizers()
        generator_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

        # The selection the predictor reads is detached, so what reaches the
        # generator here is the shared criteria and nothing else. The two
        # optimizers hold disjoint parameters, so the order they step in does
        # not matter.
        predictor_total, predictor_losses, output = self.predictor_phase_loss(batch)
        self.manual_backward(predictor_total)
        generator_optimizer.step()
        predictor_optimizer.step()
        generator_optimizer.zero_grad()
        predictor_optimizer.zero_grad()

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
                    f"{self.predictor_phase}_{name}": value
                    for name, value in predictor_losses.items()
                },
                **{
                    f"generator_{name}": value
                    for name, value in generator_losses.items()
                },
            },
        )
        return total
