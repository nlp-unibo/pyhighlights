"""Task registrations: a corpus, a model, and what to score them with.

The metric set follows the corpus. Beer, Hotel, Movies and Toy have two
classes and take the two-class accuracy and F1. A task over HateXplain, which
has three, names ``MULTICLASS_ACCURACY_METRIC`` and ``MULTICLASS_F1_METRIC``
instead. Highlight and selection metrics are the same everywhere, since a
token is either selected or it is not.
"""

from typing import TYPE_CHECKING, Any, Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.configurations.keys import (
    ACCURACY_METRIC,
    CLASS_WEIGHTS,
    F1_METRIC,
    GRU_FR,
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    HIGHLIGHT_LOSS,
    HIGHLIGHT_PRECISION_METRIC,
    HIGHLIGHT_RECALL_METRIC,
    LOSS_CHECKPOINT,
    LOSS_EARLY_STOPPING,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
    SELECTION_SPANS_METRIC,
    TOY,
    TOY_CLASS_WEIGHTS_TASK,
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
    TOY_TASK,
)
from pyhighlights.utility.spaces import token_spaces

if TYPE_CHECKING:
    import lightning.pytorch.callbacks

    import pyhighlights.components.loaders
    import pyhighlights.components.models.base
    import pyhighlights.components.models.spp.genspp
    import pyhighlights.components.preprocessors
    import pyhighlights.utility.losses
    import pyhighlights.utility.metrics

#: What every select-then-predict run reports, whatever the corpus: how much
#: of the document was kept, where it sits, and how well what was kept matches
#: the annotation. Precision and recall are reported beside their F1 because a
#: sparsity target moves them apart, and spans because a size alone does not
#: say whether the kept words sit together.
HIGHLIGHT_METRICS = [
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    HIGHLIGHT_PRECISION_METRIC,
    HIGHLIGHT_RECALL_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
    SELECTION_SPANS_METRIC,
]

#: Binary corpora: Beer, Hotel, Movies, Toy.
BINARY_METRICS = [ACCURACY_METRIC, F1_METRIC, *HIGHLIGHT_METRICS]


def pretrained(model: Any) -> bool:
    """Whether the encoders of ``model`` read a pretrained subword vocabulary."""
    return model is not None and bool(token_spaces(model) - {None})


class TaskConfig(Configuration):
    """Fields every task shares.

    Anything decided before a run is declared here, so a key records what was
    asked for rather than the part of it somebody remembered to register. A
    value a run computes cannot be declared at all: the embedding matrix a
    vector file is read into is fitted against the training split at run time,
    so it reaches the model as a tensor and never as a parameter.
    """

    name: str = Param(
        "task",
        description="Name of the task, which names the directory its results go to.",
    )
    save_path: str | None = Param(
        None,
        description=(
            "Directory the results are written under, or None for ``results`` "
            "in the working directory."
        ),
    )
    seeds: Sequence[int] = Param(
        [42],
        description=(
            "Random seeds the task runs with, training one model from scratch per seed."
        ),
    )
    batch_size: int = Param(32, ge=1, description="Number of examples per batch.")
    max_length: int | None = Param(
        None,
        description=(
            "Maximum number of encoded positions per example, special tokens "
            "included, or None to keep the whole input."
        ),
    )
    vocabulary_size: int = Param(
        10_000,
        ge=3,
        description=(
            "Size of the vocabulary the task fits on the training split, which "
            "has to match the backbone's ``vocabulary_size``."
        ),
    )
    add_special_tokens: bool = Param(
        True,
        description=(
            "Whether a pretrained tokenizer keeps ``[CLS]`` and ``[SEP]``, which "
            "the encoder was trained reading. They carry no word, so a selector "
            "never sees them either way."
        ),
    )
    embeddings: str | None = Param(
        None,
        description=(
            "Path to a GloVe-style vector file the tokens are embedded from, or "
            "None for none."
        ),
    )
    pretrained_tokens_only: bool = Param(
        True,
        description=(
            "Whether a token the vector file has no vector for is dropped and "
            "read as the unknown token. False gives it a randomly initialized "
            "row of its own."
        ),
    )
    vocabulary_from: str = Param(
        "corpus",
        description=(
            "Where the vocabulary comes from: ``corpus`` fits it on the training "
            "split and looks its tokens up in the vector file, ``splits`` does "
            "the same over every split, and ``vectors`` takes the file's own "
            "vocabulary whole. The last two need ``embeddings`` and refuse to "
            "build without them."
        ),
    )
    requires_embeddings: bool = Param(
        False,
        description=(
            "Whether the task refuses to build without ``embeddings``, for a "
            "reproduction whose numbers come from a released vector file. It is "
            "implied when ``vocabulary_from`` is ``splits`` or ``vectors``."
        ),
    )
    one_hot_embeddings: int | None = Param(
        None,
        ge=1,
        description=(
            "Width of a one-hot embedding table built in place of a read or "
            "learned one, for a corpus whose tokens are symbols. None builds no "
            "such table."
        ),
    )
    callbacks: List[RegistrationKey["lightning.pytorch.callbacks.Callback"]] = Param(
        [LOSS_EARLY_STOPPING, LOSS_CHECKPOINT],
        description=(
            "Registration keys of the callbacks that monitor the run: early "
            "stopping, checkpointing, and any criterion they read, such as the "
            "``GeneralizationLossScore`` that the ``SCORE_*`` callbacks monitor. "
            "The stopping and checkpoint callbacks have to monitor the same "
            "quantity, or the run reports a model its own stopping rule did not "
            "choose."
        ),
    )
    store_predictions: bool = Param(
        False,
        description=(
            "Whether the test predictions of each seed are written to "
            "``predictions-seed=<seed>.pkl`` in the run directory."
        ),
    )
    keep_checkpoints: bool = Param(
        True,
        description=(
            "Whether the checkpoints survive the run, which nothing downstream "
            "needs because the task restores the best one itself before scoring. "
            "Turning this off trades re-scoring without retraining for disk "
            "space, which a grid of transformer cells runs out of."
        ),
    )
    save_weights_only: bool = Param(
        False,
        description=(
            "Whether a checkpoint holds the weights without the optimizer state. "
            "That state is most of a fine-tuned encoder's file and is needed "
            "only to resume training, which no task does."
        ),
    )
    faithfulness: bool = Param(
        False,
        description=(
            "Whether the faithfulness terms are scored on the test split beside "
            "the metrics."
        ),
    )
    diagnostics: bool = Param(
        False,
        description=(
            "Whether the run records what each pipeline stage held, which is "
            "allowed only for a run whose batches are bounded."
        ),
    )
    highlight_supervision: bool = Param(
        False,
        description=(
            "Whether the model is also trained against the highlight annotation "
            "through ``highlight_loss``, which needs an annotated training split."
        ),
    )
    highlight_loss: RegistrationKey["pyhighlights.utility.losses.Loss"] = Param(
        HIGHLIGHT_LOSS,
        description=(
            "Registration key of the loss that scores the selection against the "
            "highlight annotation under ``highlight_supervision``."
        ),
    )
    highlight_coefficient: float = Param(
        1.0,
        ge=0.0,
        description="Non-negative weight of ``highlight_loss`` in the total loss.",
    )
    trainer_args: Dict[str, Any] = Param(
        {"accelerator": "cpu", "max_epochs": 5},
        description=(
            "Keyword arguments for the Lightning trainer, applied over the "
            "defaults of the task."
        ),
    )

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
                "A task embeds its tokens with the pretrained encoder of its "
                "model, a vector file or a one-hot table, never with two of "
                "them."
            ),
            condition=lambda task: (
                pretrained(getattr(task, "model", None))
                + (task.embeddings is not None)
                + (task.one_hot_embeddings is not None)
                <= 1
            ),
        )
        return config


@register_class(
    key=TOY_TASK, component="pyhighlights.components.tasks.SPPTask", run_method="run"
)
class ToyTaskConfig(TaskConfig):
    """The synthetic corpus against FR: a sanity check run that trains in seconds."""

    name: str = Param("toy")
    loader: RegistrationKey["pyhighlights.components.loaders.HighlightLoader"] = Param(
        TOY, description="Registration key of the loader that reads the corpus."
    )
    model: RegistrationKey["pyhighlights.components.models.base.Model"] = Param(
        GRU_FR, description="Registration key of the model the task trains."
    )
    preprocessor: (
        RegistrationKey["pyhighlights.components.preprocessors.Preprocessor"] | None
    ) = Param(
        None,
        description=(
            "Registration key of the preprocessor applied to the corpus, or None "
            "for a corpus that needs none."
        ),
    )
    train_metrics: List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] = (
        Param(
            BINARY_METRICS,
            description=(
                "Registration keys of the metrics scored on the training split."
            ),
        )
    )
    val_metrics: List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] = (
        Param(
            BINARY_METRICS,
            description=(
                "Registration keys of the metrics scored on the validation split."
            ),
        )
    )
    test_metrics: List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] = (
        Param(
            BINARY_METRICS,
            description=("Registration keys of the metrics scored on the test split."),
        )
    )
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

    name: str = Param(
        "class-weights",
        description="Name of the task, which names the directory its results go to.",
    )
    loader: RegistrationKey["pyhighlights.components.loaders.HighlightLoader"] = Param(
        TOY,
        description="Registration key of the loader that reads the corpus to weigh.",
    )
    weights: RegistrationKey["pyhighlights.components.preprocessors.ClassWeights"] = (
        Param(
            CLASS_WEIGHTS,
            description=(
                "Registration key of the preprocessor that computes the class weights."
            ),
        )
    )
    preprocessor: (
        RegistrationKey["pyhighlights.components.preprocessors.Preprocessor"] | None
    ) = Param(
        None,
        description=(
            "Registration key of the preprocessor applied to the corpus before it "
            "is weighed, or None for a corpus that needs none."
        ),
    )
    save_path: str | None = Param(
        None,
        description=(
            "Directory the results are written under, or None for ``results`` "
            "in the working directory."
        ),
    )


@register_class(
    key=TOY_CLASS_WEIGHTS_TASK,
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
    result nobody could read. Nothing scores the training split, because no
    epoch of the winning model is ever trained, so only validation and test
    carry metrics.
    """

    loader: RegistrationKey["pyhighlights.components.loaders.HighlightLoader"] = Param(
        TOY, description="Registration key of the loader that reads the corpus."
    )
    search: RegistrationKey[
        "pyhighlights.components.models.spp.genspp.GenSPPTrainer"
    ] = Param(
        TOY_GENSPP_TRAINER,
        description=(
            "Registration key of the GenSPP search, which also names the model "
            "the task evolves."
        ),
    )
    preprocessor: (
        RegistrationKey["pyhighlights.components.preprocessors.Preprocessor"] | None
    ) = Param(
        None,
        description=(
            "Registration key of the preprocessor applied to the corpus, or None "
            "for a corpus that needs none."
        ),
    )
    val_metrics: List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] = (
        Param(
            BINARY_METRICS,
            description=(
                "Registration keys of the metrics scored on the validation split."
            ),
        )
    )
    test_metrics: List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] = (
        Param(
            BINARY_METRICS,
            description=("Registration keys of the metrics scored on the test split."),
        )
    )


@register_class(
    key=TOY_GENSPP_TASK,
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
