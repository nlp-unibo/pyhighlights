from __future__ import annotations

import logging
from itertools import islice
from typing import Dict, List, Set, Tuple

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.base import (
    SPP,
    SPPBackbone,
    SPPPredictor,
)
from pyhighlights.components.models.spp.data import SPPOutput
from pyhighlights.utility import diagnostics
from pyhighlights.utility.losses import Loss, compute_losses

logger = logging.getLogger(__name__)


class DAR(SPP):
    """Rationalizer whose highlight has to read like the input it came from.

    A cooperative game lets the pair agree on a private code. The selection
    drifts away from the semantics of the full input, and the predictor learns
    to read the drift. Accuracy stays high, and the generator is rewarded for a
    highlight nobody else can interpret. The paper calls that rationale shift.
    DAR answers it with a second predictor, the aligner, trained on the full
    input alone and then frozen. That module never sees a highlight during its
    own training, so it can only read one the way it reads text. Asking it to
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
    epoch. Under data parallelism every process averages its gradients with
    the others', so all processes hold the same aligner. An ``EarlyStopping``
    callback counts epochs of the *rationalizer* only, so a long pretraining
    stays off the patience counter.

    The frozen aligner stays in evaluation mode, so the alignment term is a
    fixed function of the highlight. The reference implementation reloads its
    aligner in training mode, so its dropout runs on every highlight the
    aligner scores. This library treats that as an error in the reference.
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
        # input while it trains and the highlight afterwards. Both passes ask
        # the same module what the label looks like from what it was given.
        self.aligner_loss = Registry.from_key(aligner_loss, expected_type=Loss)
        self.losses.append(self.aligner_loss)
        # Frozen from construction, and trainable only inside the pretraining
        # loop. A data-parallel wrapper registers every parameter that requires
        # a gradient when it wraps the model, and raises once one stops
        # receiving it. A frozen pretrained encoder stays frozen throughout.
        self._aligner_trainable = [
            parameter
            for parameter in self.aligner_parameters()
            if parameter.requires_grad
        ]
        for parameter in self._aligner_trainable:
            parameter.requires_grad_(False)
        # Saved with the weights, so a resumed run does not pretrain an
        # aligner the checkpoint already carries trained.
        self.register_buffer("aligner_ready", th.zeros((), dtype=th.bool))

    def aligner_parameters(self) -> List[th.nn.Parameter]:
        return [*self.aligner_backbone.parameters(), *self.aligner.parameters()]

    def encoder_ids(self) -> Set[int]:
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

    def train(self, mode: bool = True) -> DAR:
        # Lightning calls this on the whole model. A trained aligner is frozen,
        # and freezing the weights does not stop dropout.
        super().train(mode)
        if bool(self.aligner_ready):
            self.aligner_backbone.eval()
            self.aligner.eval()
        return self

    def pretrain_aligner(self) -> None:
        """Train the aligner on the full input, then freeze it.

        Once per fit and before the first epoch, so every rationalization
        batch is scored against the same module. An aligner that kept moving
        could co-adapt to the highlight, which is the thing it exists not to
        do.
        """
        loader = self.trainer.train_dataloader
        if loader is None:
            raise RuntimeError("DAR pretrains its aligner on the training loader")
        for parameter in self._aligner_trainable:
            parameter.requires_grad_(True)
        optimizer = self.build_optimizer([(self._aligner_trainable, 1.0)])
        for epoch in range(self.pretrain_epochs):
            total = 0.0
            batches = 0
            # A distributed sampler shards by epoch and is told which one by
            # the loop that owns it. This loop owns these epochs, so without
            # this every one of them is the same shard in the same order.
            sampler = getattr(loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            # The fit loop's own bound on an epoch, which is what
            # `fast_dev_run` and `limit_train_batches` set. This loop is the
            # model's rather than the loop's, so nothing else applies it: a
            # run bounded to two batches pretrained on the whole split.
            limit = self.trainer.num_training_batches
            epoch_batches = (
                loader if limit == float("inf") else islice(loader, int(limit))
            )
            diagnostics.record("phase", name="aligner", epoch=epoch)
            for batch in epoch_batches:
                batch = self.transfer_batch_to_device(batch, self.device, 0)
                optimizer.zero_grad()
                loss, _ = compute_losses(
                    [self.aligner_loss],
                    {**batch.as_dict(), "aligner_class_logits": self.align_full(batch)},
                )
                loss.backward()
                # The wrapper that averages gradients across processes does
                # not see this loop, so the loop averages them itself. One
                # process reduces to the identity.
                for parameter in self._aligner_trainable:
                    if parameter.grad is not None:
                        parameter.grad = self.trainer.strategy.reduce(
                            parameter.grad, reduce_op="mean"
                        )
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
        optimizer.zero_grad()
        for parameter in self._aligner_trainable:
            parameter.requires_grad_(False)
        self.aligner_ready.fill_(True)

    def on_train_start(self) -> None:
        super().on_train_start()
        # An aligner restored from a checkpoint, or pretrained earlier in this
        # process, is trained. Training it again on a model whose generator has
        # moved would make it a different module.
        if not bool(self.aligner_ready):
            self.pretrain_aligner()
        self.train(self.training)

    def compute_loss(
        self, input_data: InputData, output_data: SPPOutput
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        highlight_mask = self.reported(output_data).highlight_mask
        values = self.head_namespace(
            input_data,
            output_data,
            aligner_class_logits=self.align(input_data, highlight_mask),
        )
        return compute_losses(self.losses, values)
