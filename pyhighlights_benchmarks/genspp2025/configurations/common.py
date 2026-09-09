"""Settings both corpora share, and the metric sets they report.

Every baseline trains the same way in the released implementation: Adam at
1e-3, batches of 64, up to 500 epochs, early stopping on validation loss after
30 worse ones, and five seeds. Only the corpus and the model change.
"""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey

from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.keys import (
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    MULTICLASS_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
)
from pyhighlights.utility.metrics import BoundMetric
from pyhighlights_benchmarks.genspp2025.configurations.keys import SEEDS

#: What the paper reports: macro F1 for the task, token F1 against the
#: annotation, and how much of the document the selector kept.
BINARY_METRICS = [
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
]

#: The toy corpus hides one of three patterns, so its task is three-way.
THREE_CLASS_METRICS = [
    MULTICLASS_F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
]


class PaperTaskConfig(Configuration):
    """How every baseline in the paper is trained."""

    loader: RegistrationKey[HighlightLoader] = Param(None)
    model: RegistrationKey[Model] = Param(None)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    seeds: Sequence[int] = Param(SEEDS)
    batch_size: int = Param(64, ge=1)
    monitor: str = Param("val_loss")
    patience: int = Param(30, ge=0)
    save_path: str | None = Param(None)
    trainer_args: Dict[str, Any] = Param(
        {"accelerator": "auto", "devices": 1, "max_epochs": 500}
    )


class PaperGenSPPTaskConfig(Configuration):
    """GenSPP's own task: a search, not five hundred epochs of descent."""

    search: RegistrationKey[GenSPPTrainer] = Param(None)
    loader: RegistrationKey[HighlightLoader] = Param(None)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    seeds: Sequence[int] = Param(SEEDS)
    batch_size: int = Param(64, ge=1)
    save_path: str | None = Param(None)
    trainer_args: Dict[str, Any] = Param({"accelerator": "auto", "devices": 1})
