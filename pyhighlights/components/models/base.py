from __future__ import annotations

import abc
from typing import Dict, Generic, List, Literal, Mapping, Tuple, TypeVar

import lightning as L
import torch as th
from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric

from pyhighlights.components.models.data import InputData, ModelData, OutputData
from pyhighlights.utility import diagnostics
from pyhighlights.utility.binding import TOTAL_LOSS, check_names
from pyhighlights.utility.losses import Loss, build_losses, compute_losses
from pyhighlights.utility.metrics import BoundMetric, build_metrics

Split = Literal["train", "val", "test"]

#: Output type a model produces; subclasses pin it, as ``SPP`` pins ``SPPOutput``.
OutputT = TypeVar("OutputT", bound=OutputData)

__all__ = ["InputData", "Model", "ModelData", "OutputT", "OutputData", "Split"]


class Model(L.LightningModule, abc.ABC, Generic[OutputT]):
    """A Lightning module whose losses and metrics bind to named fields.

    A subclass implements :meth:`forward`, which reads a batch passed as
    ``data`` and returns an ``OutputT``. The base class does everything else a
    step needs. Each split runs its own forward method, so a model can score
    evaluation differently from training. The losses and the metrics read
    their inputs by name out of :meth:`namespace`. A model that drives its own
    optimizers calls :meth:`record` once it knows the loss of a step.

    Each split logs its summed loss, each loss term and each metric under
    ``{split}_{name}``. :meth:`setup` refuses any two of them sharing a name.
    """

    def __init__(
        self,
        name: str,
        losses: List[RegistrationKey[Loss]],
        optimizer: RegistrationKey[th.optim.Optimizer],
        train_metrics: List[RegistrationKey[BoundMetric]] | None = None,
        val_metrics: List[RegistrationKey[BoundMetric]] | None = None,
        test_metrics: List[RegistrationKey[BoundMetric]] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hyperparameters())
        self.name = name
        self.optimizer = optimizer

        self.train_metrics = build_metrics(train_metrics)
        self.val_metrics = build_metrics(val_metrics)
        self.test_metrics = build_metrics(test_metrics)
        self.losses = th.nn.ModuleList(build_losses(keys=losses))

        self.store_predictions = False
        self.predictions = []
        self.described_metrics = set()

    def ignore_hyperparameters(self) -> List[str]:
        """Constructor arguments the checkpoint leaves out.

        The metrics are left out because they score a model without defining
        it. A task can also replace them after construction, so the stored
        keys could name metrics the model never ran.
        """
        return ["train_metrics", "val_metrics", "test_metrics"]

    @abc.abstractmethod
    def forward(self, data: InputData) -> OutputT: ...

    def loss_names(self) -> List[str]:
        """Every name a step of this model logs a loss term under.

        A model that logs terms beyond :attr:`losses`, such as a second loss
        list or a per-phase prefix, extends this list so that :meth:`setup`
        checks those names too.
        """
        return [loss.name for loss in self.losses]

    def setup(self, stage: str) -> None:
        # Here rather than in the constructor, because a task can replace the
        # metrics after construction. Lightning calls this before any loop.
        losses = self.loss_names()
        for split in ("train", "val", "test"):
            check_names(losses, [m.name for m in getattr(self, f"{split}_metrics")])

    def namespace(
        self, input_data: InputData, output_data: OutputData, **extra: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """Fields losses and metrics can bind to, latest definition winning."""
        return {**input_data.as_dict(), **output_data.as_dict(), **extra}

    def load_knowledge(self, data: InputData) -> None:
        """Adopt the corpus's knowledge base, already tokenized.

        A model that classifies from the input alone does not implement this
        method. It refuses a knowledge base rather than accepting one and
        ignoring it.

        The knowledge base arrives as an :class:`InputData` of ``M`` rows,
        because the collator is what turns text into tensors. Its ``y_true``
        carries nothing, since a knowledge base entry has no label.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not read a knowledge base"
        )

    def faithfulness(
        self, input_data: InputData, output_data: OutputT
    ) -> Dict[str, th.Tensor]:
        """Per-sample faithfulness terms, when the architecture defines them.

        The terms need the predictor run against masks of their own, which is
        a property of a model family rather than of every model. A model that
        cannot produce them raises, so that a task asked for faithfulness
        fails rather than reporting a column it never measured.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not measure faithfulness"
        )

    def update_metrics(self, split: Split, input_data: InputData, output_data: OutputT):
        values = self.namespace(input_data, output_data)
        metrics = getattr(self, f"{split}_metrics")
        # The first call records the fields a metric can bind to and the
        # fields each metric reads. Neither changes between batches, so each
        # split records them once.
        if split not in self.described_metrics and diagnostics.active():
            self.described_metrics.add(split)
            diagnostics.record("metric", split=split, namespace=sorted(values))
            diagnostics.record(
                "metric", **{metric.name: metric.inputs for metric in metrics}
            )
        for metric in metrics:
            metric.update(values)

    def configure_optimizers(self):
        return Registry.from_key(self.optimizer, params=self.parameters())

    def log_values(
        self,
        split: Split,
        values: Mapping[str, th.Tensor | Metric],
        batch_size: int,
    ) -> None:
        """Log losses and metrics as epoch values under ``{split}_{name}``.

        Lightning averages a tensor over the epoch, weighted by
        ``batch_size``, and over every process: under data parallelism each
        process reads its own shard, and a monitor reading one process's loss
        would decide on that shard alone. A ``Metric`` synchronises its own
        state, and is computed and reset by Lightning at the end of the epoch.
        """
        for name, value in values.items():
            self.log(
                name=f"{split}_{name}",
                value=value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
                sync_dist=True,
            )

    def training_forward(self, batch: InputData) -> OutputT:
        return self.forward(data=batch)

    def validation_forward(self, batch: InputData) -> OutputT:
        return self.training_forward(batch=batch)

    def test_forward(self, batch: InputData) -> OutputT:
        return self.training_forward(batch=batch)

    def record(
        self,
        split: Split,
        batch: InputData,
        output_data: OutputT,
        total_loss: th.Tensor,
        losses: Dict[str, th.Tensor],
    ) -> None:
        """Update the metrics, log them with the losses, keep the predictions.

        Every step does this once its loss is known. A model driving its own
        optimizers computes that loss in phases and calls this afterwards.
        """
        self.update_metrics(split=split, input_data=batch, output_data=output_data)
        metrics = {m.name: m.metric for m in getattr(self, f"{split}_metrics")}
        self.log_values(
            split=split,
            values={TOTAL_LOSS: total_loss, **losses, **metrics},
            batch_size=batch.y_true.shape[0],
        )
        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output_data.as_numpy()})

    def _step(self, batch: InputData, batch_idx: int, split: Split) -> th.Tensor:
        # Every stage below reports per batch without knowing its split, so
        # this line says which split and batch the following lines belong to.
        diagnostics.record("step", split=split, batch=batch_idx)
        forward = {
            "train": self.training_forward,
            "val": self.validation_forward,
            "test": self.test_forward,
        }[split]
        output_data = forward(batch)
        total_loss, losses = self.compute_loss(
            input_data=batch, output_data=output_data
        )
        self.record(split, batch, output_data, total_loss, losses)
        return total_loss

    def training_step(self, batch: InputData, batch_idx: int):
        return self._step(batch=batch, batch_idx=batch_idx, split="train")

    def validation_step(self, batch: InputData, batch_idx: int):
        return self._step(batch=batch, batch_idx=batch_idx, split="val")

    def test_step(self, batch: InputData, batch_idx: int):
        return self._step(batch=batch, batch_idx=batch_idx, split="test")

    def compute_loss(
        self,
        input_data: InputData,
        output_data: OutputT,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        return compute_losses(self.losses, self.namespace(input_data, output_data))
