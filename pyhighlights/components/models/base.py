import abc
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import lightning as L
import torch as th
from cinnamon.registry import RegistrationKey
from torchmetrics import Metric, MetricCollection

from pyhighlights.utility.losses import Loss, build_losses
from pyhighlights.utility.metrics import build_torchmetrics

Split = Literal["train", "val", "test"]

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass
class ModelData:
    def as_numpy(self):
        return {
            key: value.detach().cpu().numpy()
            for key, value in self.__dict__.items()
            if isinstance(value, th.Tensor)
        }


@dataclass
class InputData(ModelData):
    features: th.Tensor
    mask: th.Tensor
    sample_ids: th.Tensor
    y_true: th.Tensor
    highlight_true: th.Tensor


@dataclass
class OutputData(ModelData):
    y_pred: th.Tensor
    highlight_logits: th.Tensor
    highlight_pred: th.Tensor


# ---------------------------------------------------------------------------
# Base model
# ---------------------------------------------------------------------------


class Model(L.LightningModule, abc.ABC):
    def __init__(
        self,
        name: str,
        losses: List[RegistrationKey[Loss]],
        learning_rate: float = 1e-03,
        train_metrics: Dict[str, RegistrationKey[Metric]] | None = None,
        val_metrics: Dict[str, RegistrationKey[Metric]] | None = None,
        test_metrics: Dict[str, RegistrationKey[Metric]] | None = None
    ):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hyperparameters())

        self.name = name
        self.learning_rate = learning_rate

        # Metrics
        self.train_metrics: MetricCollection | None = None
        if train_metrics is not None:
            self.train_metrics = build_torchmetrics(train_metrics)

        self.val_metrics: MetricCollection | None = None
        if val_metrics is not None:
            self.val_metrics = build_torchmetrics(val_metrics)

        self.test_metrics: MetricCollection | None = None
        if test_metrics is not None:
            self.test_metrics = build_torchmetrics(test_metrics)

        # Losses
        self.losses = build_losses(keys=losses)

        # Misc

        # Support variable to store predictions during evaluate() or test()
        self.store_predictions = False
        self.predictions = []

        self.forward_mapping = {
            "train": self.training_forward,
            "val": self.validation_forward,
            "test": self.test_forward,
        }

    def ignore_hyperparameters(self) -> List[str]: ...

    def enable_storing_predictions(self):
        self.store_predictions = True

    def disable_storing_predictions(self):
        self.store_predictions = False

    def flush_predictions(self):
        self.predictions.clear()

    def update_metrics(
        self, split: Split, input_data: InputData, output_data: OutputData
    ):
        # input_data.features:  [bs, F]
        # input_data.mask:      [bs, F]
        # input_data.y_true:    [bs,]
        #
        # output_data.y_pred:   [bs, C]

        metrics: MetricCollection = getattr(self, f"{split}_metrics")
        if metrics is None:
            return

        metrics.update(input_data, output_data)

    def compute_metrics(self, split: Split):
        metrics: MetricCollection = getattr(self, f"{split}_metrics")
        if metrics is None:
            return

        metric_values = metrics.compute()
        for key, value in metric_values.items():
            self.log(f"{split}_{key}", value, prog_bar=True)
        metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.compute_classification_metrics(split="val")

    def on_test_epoch_end(self) -> None:
        self.compute_classification_metrics(split="test")

    def configure_optimizers(self):
        return th.optim.AdamW(self.parameters(), lr=self.learning_rate)

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
        forward_method = self.forward_mapping[split]
        output_data = forward_method(data=batch)

        batch_size = output_data.y_pred.shape[0]

        total_loss, losses = self.compute_loss(
            input_data=batch, output_data=output_data
        )

        self.log_metrics(
            split=split, total_loss=total_loss, losses=losses, batch_size=batch_size
        )

        self.update_classification_metrics(
            split=split, input_data=batch, output_data=output_data
        )

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
        total_loss = th.Tensor(0.0, device=input_data.y_true.device)
        losses = {}

        for loss in self.losses:
            if not loss.enabled:
                continue

            loss_value = loss.forward(input_data=input_data, output_data=output_data)
            total_loss += loss_value * loss.coefficient
            losses[loss.name] = loss_value

        return total_loss, losses
