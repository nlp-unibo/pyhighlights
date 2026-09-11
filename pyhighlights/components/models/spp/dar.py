import logging
from typing import Dict, List, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
)
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility.losses import Loss, compute_losses

logger = logging.getLogger(__name__)


class DAR(SPP):
    """Rationalizer whose highlight has to read like the input it came from.

    A cooperative game lets the pair agree on a private code: the selection
    drifts away from the semantics of the full input, the predictor learns to
    read the drift, accuracy stays high and the generator is rewarded for a
    highlight nobody else can interpret. The paper calls that rationale shift.
    DAR answers it with a second predictor -- an aligner -- trained on the full
    input alone and then frozen. That module never sees a highlight during its
    own training, so it can only read one the way it reads text; asking it to
    predict the label *from the highlight* therefore costs the generator
    anything it selected in a private code. The aligner is frozen, so this term
    trains the generator only.

    Liu, Wang, Wang, Deng, Zhang, Wang and Li, 2024, *Enhancing the
    Rationale-Input Alignment for Self-explaining Rationalization*, ICDE 2024,
    pages 2218-2230.
    Paper: <https://doi.org/10.1109/ICDE60146.2024.00176>.
    Reference implementation: <https://github.com/jugechengzi/dar>.

    The aligner is pretrained here rather than by the task: it is part of the
    model, is checkpointed with it, and a resumed run finds it trained. The
    pretraining runs its own loop over the training loader before the first
    epoch, which is why two limits are worth knowing. Under data parallelism
    each process pretrains its own aligner on its own shard, with no gradient
    sync -- the loop is outside the strategy Lightning drives. And an
    ``EarlyStopping`` callback counts epochs of the *rationalizer*, since the
    pretraining is not one of them; the reference implementation spends 100
    epochs there, and this keeps that off the patience counter.
    """

    def __init__(
        self,
        aligner_backbone: RegistrationKey[SPPBackbone],
        predictor: RegistrationKey[SPPPredictor],
        aligner_loss: RegistrationKey[Loss],
        pretrain_epochs: int = 20,
        **kwargs,
    ):
        super().__init__(predictor=predictor, **kwargs)
        if len(self.selectors) != 1:
            raise ValueError("DAR requires exactly one selector")
        if pretrain_epochs < 1:
            raise ValueError(
                "DAR needs at least one pretraining epoch: an aligner that "
                "never read the full input measures no alignment to it"
            )

        self.aligner_backbone = Registry.from_key(
            aligner_backbone, expected_type=SPPBackbone
        )
        self.aligner = Registry.from_key(
            predictor,
            expected_type=SPPPredictor,
            input_size=self.aligner_backbone.output_size,
        )
        self.pretrain_epochs = pretrain_epochs
        # The same binding scores both passes: the aligner reads the full
        # input while it trains and the highlight afterwards, and both are the
        # same question asked of the same module -- what the label looks like
        # from what it was given.
        self.aligner_loss = Registry.from_key(aligner_loss, expected_type=Loss)
        self.losses.append(self.aligner_loss)
        # Saved with the weights, so a resumed run does not pretrain an
        # aligner the checkpoint already carries trained.
        self.register_buffer("aligner_ready", th.zeros((), dtype=th.bool))

    def aligner_parameters(self) -> List[th.nn.Parameter]:
        return [*self.aligner_backbone.parameters(), *self.aligner.parameters()]

    def encoder_ids(self) -> set:
        return super().encoder_ids() | {
            id(parameter) for parameter in self.aligner_backbone.parameters()
        }

    def load_embeddings(self, matrix: th.Tensor) -> None:
        super().load_embeddings(matrix)
        self.aligner_backbone.load_embeddings(matrix)

    def configure_optimizers(self):
        # The aligner is left out: it is trained once, before the first epoch,
        # and frozen for the whole of what this optimizer drives.
        aligner = {id(parameter) for parameter in self.aligner_parameters()}
        return self.build_optimizer(
            [
                (
                    [
                        parameter
                        for parameter in self.parameters()
                        if id(parameter) not in aligner
                    ],
                    1.0,
                )
            ]
        )

    def align(self, data: InputData, highlight_mask: th.Tensor) -> th.Tensor:
        """What the aligner makes of a selection."""
        return self.predict(
            data=data,
            highlight_mask=highlight_mask,
            backbone=self.aligner_backbone,
            predictor=self.aligner,
        )

    def align_full(self, data: InputData) -> th.Tensor:
        """What the aligner makes of the whole input, which is all it is taught."""
        return self.align(data, th.ones_like(self.selection_valid(data)))

    def pretrain_aligner(self) -> None:
        """Train the aligner on the full input, then freeze it.

        Once per fit and before the first epoch, so every rationalization
        batch is scored against the same module -- an aligner that kept moving
        could co-adapt to the highlight, which is the thing it exists not to
        do.
        """
        loader = self.trainer.train_dataloader
        if loader is None:
            raise RuntimeError("DAR pretrains its aligner on the training loader")
        optimizer = self.build_optimizer([(self.aligner_parameters(), 1.0)])
        for epoch in range(self.pretrain_epochs):
            total = 0.0
            batches = 0
            # A distributed sampler shards by epoch and is told which one by
            # the loop that owns it. This loop owns these epochs, so without
            # this every one of them is the same shard in the same order.
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for batch in loader:
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                optimizer.zero_grad()
                loss, _ = compute_losses(
                    [self.aligner_loss],
                    {**batch.as_dict(), "aligner_class_logits": self.align_full(batch)},
                )
                loss.backward()
                optimizer.step()
                total += loss.item()
                batches += 1
            logger.info(
                "%s: aligner pretraining epoch %s/%s, loss %.4f",
                self.name,
                epoch + 1,
                self.pretrain_epochs,
                total / max(batches, 1),
            )
        for parameter in self.aligner_parameters():
            parameter.requires_grad_(False)
        self.aligner_ready.fill_(True)

    def on_train_start(self) -> None:
        super().on_train_start()
        if bool(self.aligner_ready):
            # Restored from a checkpoint, or already pretrained in this
            # process: either way it is trained, and training it again on a
            # model whose generator has moved is a different module.
            for parameter in self.aligner_parameters():
                parameter.requires_grad_(False)
            return
        self.pretrain_aligner()

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        if output_data.class_logits.shape[1] != 1:
            raise ValueError("DAR output must contain exactly one head")
        highlight_mask = next(output_data.unbind(dim=1)).highlight_mask
        values = self.head_namespace(
            input_data,
            output_data,
            aligner_class_logits=self.align(input_data, highlight_mask),
        )
        return compute_losses(self.losses, values)
