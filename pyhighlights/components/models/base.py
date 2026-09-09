from __future__ import annotations

import abc
from typing import Dict, Generic, List, Literal, Tuple, TypeVar

import lightning as L
import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.data import InputData, ModelData, OutputData
from pyhighlights.utility.losses import Loss, build_losses, compute_losses
from pyhighlights.utility.metrics import BoundMetric, build_metrics

Split = Literal["train", "val", "test"]

#: Output type a model produces; subclasses pin it, as ``SPP`` pins ``SPPOutput``.
OutputT = TypeVar("OutputT", bound=OutputData)

__all__ = ["InputData", "Model", "ModelData", "OutputT", "OutputData", "Split"]


class Model(L.LightningModule, abc.ABC, Generic[OutputT]):
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
        self.forward_mapping = {
            "train": self.training_forward,
            "val": self.validation_forward,
            "test": self.test_forward,
        }

    def ignore_hyperparameters(self) -> List[str]:
        return []

    def enable_storing_predictions(self):
        self.store_predictions = True

    def disable_storing_predictions(self):
        self.store_predictions = False

    def flush_predictions(self):
        self.predictions.clear()

    def namespace(
        self, input_data: InputData, output_data: OutputData, **extra: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """Fields losses and metrics can bind to, latest definition winning."""
        return {**input_data.as_dict(), **output_data.as_dict(), **extra}

    def faithfulness(
        self, input_data: InputData, output_data: OutputT
    ) -> Dict[str, th.Tensor]:
        """Per-sample faithfulness terms, when the architecture defines them.

        Optional, like :meth:`load_embeddings` on a backbone: the terms need
        the predictor run against masks of their own, which is a property of
        how a family of models is put together rather than of every model.
        A model that cannot produce them says so, so that a task asked for
        faithfulness fails rather than reporting a column it never measured.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not measure faithfulness"
        )

    def update_metrics(self, split: Split, input_data: InputData, output_data: OutputT):
        values = self.namespace(input_data, output_data)
        for metric in getattr(self, f"{split}_metrics"):
            metric.update(values)

    def compute_metrics(self, split: Split):
        for metric in getattr(self, f"{split}_metrics"):
            self.log(f"{split}_{metric.name}", metric.compute(), prog_bar=True)
            metric.reset()

    def on_train_epoch_end(self) -> None:
        self.compute_metrics(split="train")

    def on_validation_epoch_end(self) -> None:
        self.compute_metrics(split="val")

    def on_test_epoch_end(self) -> None:
        self.compute_metrics(split="test")

    def configure_optimizers(self):
        return Registry.from_key(self.optimizer, params=self.parameters())

    def log_metrics(
        self,
        split: Split,
        total_loss: th.Tensor,
        losses: Dict[str, th.Tensor],
        batch_size: int,
    ):
        self.log(
            name=f"{split}_loss",
            value=total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        for loss_name, loss_value in losses.items():
            self.log(
                name=f"{split}_{loss_name}",
                value=loss_value,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                batch_size=batch_size,
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
        """Log the losses, update the metrics, keep the predictions if asked.

        What every step does once its loss is known, however it got there: a
        model driving its own optimizers computes that loss in phases, but has
        the same record to write afterwards.
        """
        self.log_metrics(
            split=split,
            total_loss=total_loss,
            losses=losses,
            batch_size=batch.y_true.shape[0],
        )
        self.update_metrics(split=split, input_data=batch, output_data=output_data)
        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output_data.as_numpy()})

    def _step(self, batch: InputData, batch_idx: int, split: Split) -> th.Tensor:
        output_data = self.forward_mapping[split](batch)
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
