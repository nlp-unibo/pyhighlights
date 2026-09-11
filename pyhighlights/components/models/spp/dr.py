from typing import Dict, List

import torch as th
from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.base import InputData, Split
from pyhighlights.components.models.spp.base import SPP, SPPBackbone
from pyhighlights.components.models.spp.data import SPPOutput


class DR(SPP):
    """Rationalizer whose predictor learns at the rate of what it is given.

    Degeneration is the predictor overfitting the uninformative text a
    not-yet-trained selector hands it, and the paper bridges that to the
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
    table, so the pair is separate throughout -- the arrangement MCD uses, and
    the reason ``predictor_backbone`` is required rather than optional.

    Two limits worth knowing. DR owns the predictor's rate, so a learning-rate
    scheduler over that group would be overwritten at the next step -- nothing
    in the library configures one. And under data parallelism each process
    scales by the selection rate of its own batch, which is what a single-GPU
    reference implementation cannot say anything about.
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
        # selector which tokens were worth keeping. The paper's floor.
        self.scale_floor = scale_floor
        self.predictor_rates: Dict[int, float] = {}
        self.batch_rates: List[float] = []

    def configure_optimizers(self):
        generator = [
            *self.selector_backbones.parameters(),
            *self.selectors.parameters(),
        ]
        predictor = [
            *self.predictor_backbone.parameters(),
            *self.predictor.parameters(),
        ]
        # One optimizer, two groups, both at the optimizer's own rate: the
        # asymmetry is written per batch rather than declared here, since it
        # is the selection rate and nobody knows that before the batch.
        optimizer = self.build_optimizer([(generator, 1.0), (predictor, 1.0)])
        # `build_optimizer` splits a group in two when `encoder_lr` is set, so
        # the predictor can hold more than one group. Each is remembered with
        # the rate it was built at, and every rescale is written from that
        # base -- scaling the current value instead would compound the factor
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

    def selection_rate(self, data: InputData, output_data: SPPOutput) -> th.Tensor:
        """What fraction of the selectable input this batch's selection kept.

        A rate, not a term: it sets an optimizer's rate and never reaches a
        loss, so it is read outside the graph.
        """
        with th.no_grad():
            head = self.aggregator(output_data)
            valid = self.selection_valid(data).to(head.highlight_mask.dtype)
            return (head.highlight_mask * valid).sum() / valid.sum().clamp_min(1)

    def on_before_optimizer_step(self, optimizer: th.optim.Optimizer) -> None:
        """Write the predictor's rate for the step about to be taken.

        Here rather than beside the forward pass because this fires once per
        *step*: under gradient accumulation several batches are selected
        before one update, and the rate that update is taken at is the mean of
        what they kept rather than whichever batch happened to be last.
        """
        if not self.predictor_rates:
            raise RuntimeError("DR needs configure_optimizers before it can train")
        if not self.batch_rates:
            return
        scale = max(
            sum(self.batch_rates) / len(self.batch_rates),
            self.scale_floor,
        )
        self.batch_rates.clear()
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
            self.batch_rates.append(self.selection_rate(batch, output_data).item())
        super().record(
            split=split,
            batch=batch,
            output_data=output_data,
            total_loss=total_loss,
            losses=losses,
        )
