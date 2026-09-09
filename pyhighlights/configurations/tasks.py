"""Task registrations: a corpus, a model, and what to score them with.

The metric set follows the corpus. Beer, Hotel, Movies and Toy are binary, so
they take the binary accuracy and F1; HateXplain has three classes and takes
the multiclass ones. Highlight and selection metrics are the same everywhere,
since a token is either selected or it is not.
"""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.configurations.keys import (
    ACCURACY_METRIC,
    F1_METRIC,
    GRU_FR,
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    HIGHLIGHT_LOSS,
    NAMESPACE,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
    TOY,
    TOY_GENSPP_TRAINER,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric

#: What every select-then-predict run reports, whatever the corpus: how much
#: of the document was kept, and how well what was kept matches the annotation.
HIGHLIGHT_METRICS = [
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
]

#: Binary corpora: Beer, Hotel, Movies, Toy.
BINARY_METRICS = [ACCURACY_METRIC, F1_METRIC, *HIGHLIGHT_METRICS]


class TaskConfig(Configuration):
    """Fields every task shares."""

    name: str = Param("task")
    save_path: str | None = Param(None)
    seeds: Sequence[int] = Param([42])
    batch_size: int = Param(32, ge=1)
    max_length: int | None = Param(None)
    highlight_supervision: bool = Param(False)
    highlight_loss: RegistrationKey[Loss] = Param(HIGHLIGHT_LOSS)
    highlight_coefficient: float = Param(1.0, ge=0.0)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu", "max_epochs": 5})


class ToyTaskConfig(TaskConfig):
    """The synthetic corpus against FR: a smoke test that trains in seconds."""

    name: str = Param("toy")
    loader: RegistrationKey[HighlightLoader] = Param(TOY)
    model: RegistrationKey[Model] = Param(GRU_FR)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    train_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    vocabulary_size: int = Param(10_000, ge=2)
    batch_size: int = Param(8, ge=1)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu", "max_epochs": 2})

    @classmethod
    @register_method(
        name="task",
        tags={"toy"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.SPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


class GenSPPTaskConfig(TaskConfig):
    """Fields a GenSPP task adds, and the one it drops.

    A GenSPP task names a search rather than a model: the model key is the
    search's own, since the two disagreeing about which model was evolved is a
    result nobody could read. Nothing scores the training split -- no epoch of
    the winning model is ever trained -- so only validation and test carry
    metrics.
    """

    loader: RegistrationKey[HighlightLoader] = Param(TOY)
    search: RegistrationKey[GenSPPTrainer] = Param(TOY_GENSPP_TRAINER)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    vocabulary_size: int = Param(10_000, ge=2)


class ToyGenSPPTaskConfig(GenSPPTaskConfig):
    """The synthetic corpus against GenSPP, searched two candidates wide."""

    name: str = Param("toy-genspp")
    batch_size: int = Param(8, ge=1)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu"})

    @classmethod
    @register_method(
        name="task",
        tags={"genspp", "toy"},
        namespace=NAMESPACE,
        component="pyhighlights.components.tasks.GenSPPTask",
        run_method="run",
    )
    def default(cls):
        return super().default()


__all__: List[str] = [
    "BINARY_METRICS",
    "HIGHLIGHT_METRICS",
    "GenSPPTaskConfig",
    "TaskConfig",
    "ToyGenSPPTaskConfig",
    "ToyTaskConfig",
]
