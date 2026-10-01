from typing import Dict, List

import torch as th
from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.base import InputData, Split
from pyhighlights.components.models.spp.base import SPP, SPPBackbone
from pyhighlights.components.models.spp.data import SPPOutput


class DR(SPP):
    """Rationalizer whose predictor learns at the rate of what it is given.

    Degeneration is the predictor overfitting the uninformative text a
    not-yet-trained selector hands it. The paper relates that to the
    predictor's Lipschitz constant: restrain the constant and the predictor
    stops memorizing a bad selection. DR restrains it by decoupling the two
    rates. The selector trains at the optimizer's own rate; the predictor
    trains at that rate times the fraction of the input the selection kept,
    recomputed from the mask every batch. A selector keeping a tenth of the
    text therefore trains its predictor ten times slower, and the restraint
    relaxes on its own as the selection settles.

    Liu, Wang, Wang, Li, Qiu, Zhang, Han and Zou, 2023, *Decoupled
    Rationalization with Asymmetric Learning Rates: A Flexible Lipschitz
    Restraint*, KDD 2023, pages 1535-1547.
    Paper: <https://doi.org/10.1145/3580305.3599299>.
    Reference implementation:
    <https://github.com/jugechengzi/Rationalization-DR>.

    The reference implementation shares one embedding table between the two
    encoders and separates everything above it. Here a backbone owns its own
    table, so the pair is separate throughout, as in MCD. That is why
    ``predictor_backbone`` is required rather than optional.

    The rate of a step counts the tokens of every batch that feeds it, on every
    process. Under data parallelism each process therefore applies the
    averaged gradient at the same rate. DR owns the predictor's rate, so a
    learning-rate scheduler over that group would be overwritten at the next
    step. Nothing in the library configures one.
    """

    def __init__(
        self,
        predictor_backbone: RegistrationKey[SPPBackbone] | None,
        scale_floor: float = 0.05,
        **kwargs,
    ):
        if predictor_backbone is None:
            raise ValueError("DR requires a separate predictor backbone")
        super().__init__(predictor_backbone=predictor_backbone, **kwargs)
        if len(self.selectors) != 1:
            raise ValueError("DR requires exactly one selector")
        if not 0 < scale_floor <= 1:
            raise ValueError("scale_floor must be in (0, 1]")
        # A selection that keeps almost nothing would otherwise stop the
        # predictor entirely, and a predictor that never moves cannot tell the
        # selector which tokens were worth keeping. The reference
        # implementation's floor.
        self.scale_floor = scale_floor
        self.predictor_rates: Dict[int, float] = {}
        #: ``[kept, valid]`` token counts of each training batch since the
        #: last optimizer step.
        self.pending_counts: List[th.Tensor] = []

    def configure_optimizers(self):
        generator = self.generator_parameters()
        predictor = self.predictor_parameters()
        # One optimizer, two groups, both at the optimizer's own rate. The
        # asymmetry is the selection rate, which is known only once a step's
        # batches are selected, so it is written per step.
        optimizer = self.build_optimizer([(generator, 1.0), (predictor, 1.0)])
        # `build_optimizer` splits a group in two when `encoder_lr` is set, so
        # the predictor can hold more than one group. Each is remembered with
        # the rate it was built at, and every rescale is written from that
        # base. Scaling the current value instead would compound the factor
        # batch after batch until the predictor stopped.
        #
        # By index rather than by reference: `Optimizer.load_state_dict`
        # replaces `param_groups` with fresh dictionaries, so a resumed run
        # would rescale objects the optimizer no longer owns. The rate kept
        # here is the one the registration was built at, which is also the
        # only place to read it: what a checkpoint restores is whatever the
        # last batch scaled the rate to.
        predictor_ids = {id(parameter) for parameter in predictor}
        self.predictor_rates = {
            index: group["lr"]
            for index, group in enumerate(optimizer.param_groups)
            if any(id(parameter) in predictor_ids for parameter in group["params"])
        }
        return optimizer

    def selection_counts(self, data: InputData, output_data: SPPOutput) -> th.Tensor:
        """``[kept, valid]``: the selectable tokens of a batch and those kept.

        Counts, not a term: they set an optimizer's rate and never reach a
        loss, so they are read outside the graph.
        """
        with th.no_grad():
            head = self.reported(output_data)
            valid = self.selection_valid(data).to(head.highlight_mask.dtype)
            return th.stack([(head.highlight_mask * valid).sum(), valid.sum()])

    def on_before_optimizer_step(self, optimizer: th.optim.Optimizer) -> None:
        """Write the predictor's rate for the step about to be taken.

        The rate is the kept tokens over the valid tokens of every batch since
        the last step, summed over every process. This fires once per step, so
        under gradient accumulation one update is taken at the rate of all
        the batches it accumulates.
        """
        if not self.predictor_rates:
            raise RuntimeError("DR needs configure_optimizers before it can train")
        if not self.pending_counts:
            return
        kept, valid = th.stack(self.pending_counts).sum(dim=0)
        self.pending_counts.clear()
        counts = th.stack([kept, valid])
        if th.distributed.is_available() and th.distributed.is_initialized():
            th.distributed.all_reduce(counts)
        kept, valid = counts.tolist()
        scale = max(kept / max(valid, 1.0), self.scale_floor)
        for index, base in self.predictor_rates.items():
            optimizer.param_groups[index]["lr"] = base * scale

    def record(
        self,
        split: Split,
        batch: InputData,
        output_data: SPPOutput,
        total_loss: th.Tensor,
        losses: Dict[str, th.Tensor],
    ) -> None:
        # The one point of a batch that has the mask. What is done with it
        # waits for `on_before_optimizer_step`, since a batch and a step are
        # not the same thing. Training only: an evaluation pass sets no rate.
        if split == "train":
            self.pending_counts.append(self.selection_counts(batch, output_data))
        super().record(
            split=split,
            batch=batch,
            output_data=output_data,
            total_loss=total_loss,
            losses=losses,
        )
