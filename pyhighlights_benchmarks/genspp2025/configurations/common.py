"""Settings both corpora share, and the metric sets they report.

Every baseline trains the same way in the released implementation: Adam at
1e-3, batches of 64, up to 500 epochs, early stopping on validation loss after
30 worse ones, and five seeds. Only the corpus and the model change.
"""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class
from lightning.pytorch.callbacks import Callback

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
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
    PAPER_CHECKPOINT,
    PAPER_EARLY_STOPPING,
    SEEDS,
)

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
    """How every baseline in the paper is trained.

    Left abstract: a corpus half fills in ``loader`` and a model half fills in
    ``model``, and everything below is the same for all of them.
    """

    #: The corpus, as distributed. What the paper does to it is
    #: ``preprocessor``'s business.
    loader: RegistrationKey[HighlightLoader] = Param(None)
    #: The architecture under test -- the one thing a row of the paper's table
    #: varies.
    model: RegistrationKey[Model] = Param(None)
    #: Filtering, label folding and vote aggregation. ``None`` for a corpus
    #: the paper takes as it comes.
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    #: Scored at the end of every epoch, and what the early stopping rule
    #: watches through ``val_loss``.
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    #: Scored once, on the epoch the checkpoint kept. These are the numbers in
    #: the table.
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    #: Five runs per cell. A single run of a select-then-predict model says
    #: very little, so the spread is part of the result.
    seeds: Sequence[int] = Param(SEEDS)
    #: Sixty-four, for every architecture and both corpora.
    batch_size: int = Param(64, ge=1)
    #: Early stopping and checkpointing on the validation loss, with the
    #: paper's patience of thirty rather than the library's five. Both monitor
    #: the same quantity, so the epoch that stops a run is the epoch it is
    #: scored on.
    callbacks: List[RegistrationKey[Callback]] = Param(
        [PAPER_EARLY_STOPPING, PAPER_CHECKPOINT]
    )
    #: Where the run writes. Left null, a benchmark tells each task to write
    #: inside its own directory.
    save_path: str | None = Param(None)
    #: Five hundred epochs is a budget, not a schedule: the early stopping
    #: rule above ends a run long before it.
    trainer_args: Dict[str, Any] = Param(
        {"accelerator": "auto", "devices": 1, "max_epochs": 500}
    )


class PaperGenSPPTaskConfig(Configuration):
    """GenSPP's own task: a search, not five hundred epochs of descent.

    The fields it shares with :class:`PaperTaskConfig` mean the same things.
    What is missing is as telling as what is here: no ``model``, because the
    search names its own and naming it twice is a way for the two to disagree;
    no ``callbacks``, because nothing is monitored across epochs when the
    generator is searched rather than trained.
    """

    #: The genetic search, which carries the model key and every setting of
    #: the search itself.
    search: RegistrationKey[GenSPPTrainer] = Param(None)
    loader: RegistrationKey[HighlightLoader] = Param(None)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    seeds: Sequence[int] = Param(SEEDS)
    batch_size: int = Param(64, ge=1)
    save_path: str | None = Param(None)
    trainer_args: Dict[str, Any] = Param({"accelerator": "auto", "devices": 1})


@register_class(
    name="callback",
    tags={"early_stopping"},
    namespace=NAMESPACE,
    component="pyhighlights.components.callbacks.WarmupEarlyStopping",
)
class PaperEarlyStoppingConfig(Configuration):
    """Thirty epochs of patience, which is the paper's and not the library's."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")
    patience: int = Param(30, ge=0)


@register_class(
    name="callback",
    tags={"checkpoint"},
    namespace=NAMESPACE,
    component="pyhighlights.components.callbacks.WarmupModelCheckpoint",
)
class PaperCheckpointConfig(Configuration):
    """The same quantity the run stops on, so the two cannot disagree."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")
