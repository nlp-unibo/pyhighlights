"""Task registrations: a corpus, a model, and what to score them with.

The metric set follows the corpus. Beer, Hotel, Movies and Toy are binary, so
they take the binary accuracy and F1; HateXplain has three classes and takes
the multiclass ones. Highlight and selection metrics are the same everywhere,
since a token is either selected or it is not.
"""

from typing import Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.models.base import Model
from pyhighlights.components.models.spp.genspp import GenSPPTrainer
from pyhighlights.components.preprocessors import ClassWeights, Preprocessor
from pyhighlights.configurations.keys import (
    ACCURACY_METRIC,
    CLASS_WEIGHTS,
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
    """Fields every task shares.

    Anything decided before a run is declared here, so a key records what was
    asked for rather than the part of it somebody remembered to register. A
    value a run computes cannot be declared at all: the embedding matrix a
    vector file is read into is fitted against the training split at run time,
    so it reaches the model as a tensor and never as a parameter.
    """

    name: str = Param("task")
    save_path: str | None = Param(None)
    seeds: Sequence[int] = Param([42])
    batch_size: int = Param(32, ge=1)
    max_length: int | None = Param(None)
    vocabulary_size: int = Param(10_000, ge=2)
    pretrained_model_card: str | None = Param(None)
    #: Keep ``[CLS]`` and ``[SEP]``. A pretrained encoder was trained reading
    #: them; they carry no word, so a selector never sees them either way.
    add_special_tokens: bool = Param(True)
    embeddings: str | None = Param(None)
    pretrained_tokens_only: bool = Param(True)
    monitor: str = Param("val_loss")
    patience: int = Param(5, ge=0)
    store_predictions: bool = Param(False)
    #: Whether the weights survive the run. A checkpoint holds the whole
    #: model, and nothing downstream reads one -- the task restores the best
    #: one itself before scoring, and an analyzer reads ``results.json`` and
    #: the stored predictions. On a grid of transformer cells that is hundreds
    #: of gigabytes; turning this off trades the ability to re-score without
    #: retraining for the ability to fit on a filesystem.
    keep_checkpoints: bool = Param(True)
    #: Weights without the optimizer state: most of a fine-tuned encoder's
    #: file, and only needed to resume training, which no task does.
    save_weights_only: bool = Param(False)
    faithfulness: bool = Param(False)
    highlight_supervision: bool = Param(False)
    highlight_loss: RegistrationKey[Loss] = Param(HIGHLIGHT_LOSS)
    highlight_coefficient: float = Param(1.0, ge=0.0)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu", "max_epochs": 5})

    @classmethod
    def default(cls):
        config = super().default()
        # Declared rather than checked in the component: the registry validates
        # conditions while it expands keys, so a grid that varies the embedding
        # source drops the impossible combination before anything trains,
        # instead of raising halfway through the sweep that reaches it.
        config.add_condition(
            name="one_embedding_source",
            description=(
                "A task embeds its tokens with a pretrained model card or with "
                "a vector file, never with both."
            ),
            condition=lambda task: (
                task.pretrained_model_card is None or task.embeddings is None
            ),
        )
        return config


@register_class(
    name="task",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.tasks.SPPTask",
    run_method="run",
)
class ToyTaskConfig(TaskConfig):
    """The synthetic corpus against FR: a smoke test that trains in seconds."""

    name: str = Param("toy")
    loader: RegistrationKey[HighlightLoader] = Param(TOY)
    model: RegistrationKey[Model] = Param(GRU_FR)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    train_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    val_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    test_metrics: List[RegistrationKey[BoundMetric]] = Param(BINARY_METRICS)
    batch_size: int = Param(8, ge=1)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu", "max_epochs": 2})


class ClassWeightsTaskConfig(Configuration):
    """A run whose whole result is the class weights of a corpus.

    Its own configuration rather than a field of ``TaskConfig``: it trains
    nothing, so seeds, metrics, batches and trainer arguments would all be
    fields nobody sets. A study registers one per corpus it weights, points it
    at the same loader and preprocessor its training tasks use, and copies the
    numbers the run writes into the loss the training tasks name.
    """

    name: str = Param("class-weights")
    loader: RegistrationKey[HighlightLoader] = Param(TOY)
    weights: RegistrationKey[ClassWeights] = Param(CLASS_WEIGHTS)
    preprocessor: RegistrationKey[Preprocessor] | None = Param(None)
    save_path: str | None = Param(None)


@register_class(
    name="task",
    tags={"class_weights", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.tasks.ClassWeightsTask",
    run_method="run",
)
class ToyClassWeightsTaskConfig(ClassWeightsTaskConfig):
    """The synthetic corpus, weighed: the end-to-end check of the above."""

    name: str = Param("toy-class-weights")


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


@register_class(
    name="task",
    tags={"genspp", "toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.tasks.GenSPPTask",
    run_method="run",
)
class ToyGenSPPTaskConfig(GenSPPTaskConfig):
    """The synthetic corpus against GenSPP, searched two candidates wide."""

    name: str = Param("toy-genspp")
    batch_size: int = Param(8, ge=1)
    trainer_args: Dict[str, Any] = Param({"accelerator": "cpu"})


__all__: List[str] = [
    "BINARY_METRICS",
    "HIGHLIGHT_METRICS",
    "ClassWeightsTaskConfig",
    "GenSPPTaskConfig",
    "TaskConfig",
    "ToyClassWeightsTaskConfig",
    "ToyGenSPPTaskConfig",
    "ToyTaskConfig",
]
