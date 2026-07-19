import abc
from typing import Any, Dict, List, Literal

import lightning as L
import torch as th
from cinnamon.component import Component
from torchmetrics import MetricCollection

from pyhighlights.utility.metrics import build_torchmetrics

Split = Literal["train", "val", "test"]


class Model(L.LightningModule, Component, abc.ABC):
    def __init__(
        self,
        name: str,
        learning_rate: float = 1e-03,
        train_classification_metrics: Dict[str, Dict[str, Any]] | None = None,
        val_classification_metrics: Dict[str, Dict[str, Any]] | None = None,
        test_classification_metrics: Dict[str, Dict[str, Any]] | None = None,
        log_metrics: bool = True,
    ):
        super().__init__()

        self.save_hyperparameters(ignore=self.ignore_hyperparameters())

        self.name = name
        self.learning_rate = learning_rate
        self.log_metrics = log_metrics

        self.train_classification_metrics: MetricCollection | None = None
        if train_classification_metrics is not None:
            self.train_classification_metrics = build_torchmetrics(
                train_classification_metrics
            )

        self.val_classification_metrics: MetricCollection | None = None
        if val_classification_metrics is not None:
            self.val_classification_metrics = build_torchmetrics(
                val_classification_metrics
            )

        self.test_classification_metrics: MetricCollection | None = None
        if test_classification_metrics is not None:
            self.test_classification_metrics = build_torchmetrics(
                test_classification_metrics
            )

        # Support variable to store predictions during evaluate() or test()
        self.store_predictions = False
        self.predictions = []

    def ignore_hyperparameters(self) -> List[str]: ...

    def enable_storing_predictions(self):
        self.store_predictions = True

    def disable_storing_predictions(self):
        self.store_predictions = False

    def flush_predictions(self):
        self.predictions.clear()

    def _epoch_end_classification_metrics(self, split: Split):
        metrics: MetricCollection = getattr(self, f"{split}_classification_metrics")
        if metrics is None:
            return

        metric_values = metrics.compute()
        for key, value in metric_values.items():
            self.log(f"{split}_{key}", value, prog_bar=self.log_metrics)
        metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self._epoch_end_classification_metrics(split="val")

    def on_test_epoch_end(self) -> None:
        self._epoch_end_classification_metrics(split="test")

    def configure_optimizers(self):
        return th.optim.AdamW(self.parameters(), lr=self.learning_rate)
