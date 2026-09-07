from __future__ import annotations

import abc
from typing import Dict, List, Literal, Tuple

import lightning as L
import torch as th
from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric, MetricCollection

from pyhighlights.components.models.data import (
    InputData,
    ModelData,
    OutputData,
    SPPOutput,
)
from pyhighlights.utility.losses import Loss, build_losses
from pyhighlights.utility.metrics import build_torchmetrics

Split = Literal["train", "val", "test"]

# Keep data containers importable from this module for compatibility.
__all__ = ["InputData", "Model", "ModelData", "OutputData", "SPPOutput", "Split"]


class Model(L.LightningModule, abc.ABC):
    def __init__(
        self,
        name: str,
        losses: List[RegistrationKey[Loss]],
        optimizer: RegistrationKey[th.optim.Optimizer],
        train_metrics: Dict[str, RegistrationKey[Metric]] | None = None,
        val_metrics: Dict[str, RegistrationKey[Metric]] | None = None,
        test_metrics: Dict[str, RegistrationKey[Metric]] | None = None,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hyperparameters())
        self.name = name
        self.optimizer = optimizer

        self.train_metrics = self._build_metrics(train_metrics)
        self.val_metrics = self._build_metrics(val_metrics)
        self.test_metrics = self._build_metrics(test_metrics)
        self.losses = th.nn.ModuleList(build_losses(keys=losses))

        self.store_predictions = False
        self.predictions = []
        self.forward_mapping = {
            "train": self.training_forward,
            "val": self.validation_forward,
            "test": self.test_forward,
        }

    @staticmethod
    def _build_metrics(
        keys: Dict[str, RegistrationKey[Metric]] | None,
    ) -> MetricCollection | None:
        return build_torchmetrics(keys) if keys is not None else None

    def ignore_hyperparameters(self) -> List[str]:
        return []

    def enable_storing_predictions(self):
        self.store_predictions = True

    def disable_storing_predictions(self):
        self.store_predictions = False

    def flush_predictions(self):
        self.predictions.clear()

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: OutputData
    ):
        metrics: MetricCollection | None = getattr(self, f"{split}_metrics")
        if metrics is not None:
            metrics.update(output_data.class_logits, input_data.y_true)

    def compute_metrics(self, split: Split):
        metrics: MetricCollection | None = getattr(self, f"{split}_metrics")
        if metrics is None:
            return

        for key, value in metrics.compute().items():
            self.log(f"{split}_{key}", value, prog_bar=True)
        metrics.reset()

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

    def training_forward(self, batch: InputData) -> OutputData:
        return self.forward(data=batch)

    def validation_forward(self, batch: InputData) -> OutputData:
        return self.training_forward(batch=batch)

    def test_forward(self, batch: InputData) -> OutputData:
        return self.training_forward(batch=batch)

    def _step(self, batch: InputData, batch_idx: int, split: Split) -> th.Tensor:
        output_data = self.forward_mapping[split](batch)
        total_loss, losses = self.compute_loss(
            input_data=batch, output_data=output_data
        )

        self.log_metrics(
            split=split,
            total_loss=total_loss,
            losses=losses,
            batch_size=batch.y_true.shape[0],
        )
        self.update_metrics(split=split, input_data=batch, output_data=output_data)

        if self.store_predictions:
            self.predictions.append({**batch.as_numpy(), **output_data.as_numpy()})

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
        output_data: OutputData,
    ) -> Tuple[th.Tensor, Dict[str, th.Tensor]]:
        total_loss = output_data.class_logits.new_zeros(())
        losses = {}

        for loss in self.losses:
            if not loss.enabled:
                continue
            loss_value = loss(input_data=input_data, output_data=output_data)
            total_loss = total_loss + loss_value * loss.coefficient
            losses[loss.name] = loss_value

        return total_loss, losses
