"""What monitors a run, as configurations rather than as task parameters.

Two sets are registered. The ``loss`` pair stops and checkpoints on
``val_loss``, minimised, and is the task's default. The ``score`` pair monitors
:class:`~pyhighlights.components.callbacks.GeneralizationLossScore`'s
combination and maximises it.

**Both callbacks of a pair monitor the same quantity**, and that is the point
of pairing them: the epoch that stops a run has to be the epoch the run is
scored on. A study is free to mix them, and gets a run whose reported model is
not the one its stopping rule chose.
"""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import (
    GENERALIZATION_LOSS_SCORE,
    LOSS_CHECKPOINT,
    LOSS_EARLY_STOPPING,
    SCORE_CHECKPOINT,
    SCORE_EARLY_STOPPING,
)

EARLY_STOPPING = "pyhighlights.components.callbacks.WarmupEarlyStopping"
CHECKPOINT = "pyhighlights.components.callbacks.WarmupModelCheckpoint"


@register_class(
    key=GENERALIZATION_LOSS_SCORE,
    component="pyhighlights.components.callbacks.GeneralizationLossScore",
)
class GeneralizationLossScoreConfig(Configuration):
    """A metric, charged for validation loss risen above its own best."""

    quality: str = Param(
        "val_f1", description="Name of the logged metric that the score rewards."
    )
    loss: str = Param(
        "val_loss",
        description=(
            "Name of the logged validation loss whose relative rise above its "
            "lowest value so far is charged against the metric."
        ),
    )
    coefficient: float = Param(
        2.0,
        ge=0.0,
        description=(
            "Metric forfeited per unit of relative loss regression, 0 or greater, "
            "set per architecture because the loss curves differ. "
            "A value of 0 monitors the metric alone and a large value monitors "
            "the loss alone."
        ),
    )
    name: str = Param(
        "val_score",
        description="Name under which the combined score is logged and monitored.",
    )


@register_class(key=LOSS_EARLY_STOPPING, component=EARLY_STOPPING)
class LossEarlyStoppingConfig(Configuration):
    """Stop when the validation loss stops falling."""

    monitor: str = Param(
        "val_loss",
        description="Name of the logged quantity that the callback monitors.",
    )
    mode: str = Param(
        "min",
        description=(
            "Whether the monitored quantity improves when it falls, ``min``, "
            "or when it rises, ``max``."
        ),
    )
    patience: int = Param(
        5,
        ge=0,
        description=(
            "Number of validation checks without improvement after which "
            "training stops, 0 or greater. Checks during the model's warmup "
            "epochs are skipped."
        ),
    )


@register_class(key=LOSS_CHECKPOINT, component=CHECKPOINT)
class LossCheckpointConfig(Configuration):
    """Keep the epoch with the lowest validation loss."""

    monitor: str = Param(
        "val_loss",
        description="Name of the logged quantity that the callback monitors.",
    )
    mode: str = Param(
        "min",
        description=(
            "Whether the monitored quantity improves when it falls, ``min``, "
            "or when it rises, ``max``."
        ),
    )


@register_class(key=SCORE_EARLY_STOPPING, component=EARLY_STOPPING)
class ScoreEarlyStoppingConfig(Configuration):
    """Stop when the combined score stops rising."""

    monitor: str = Param(
        "val_score",
        description="Name of the logged quantity that the callback monitors.",
    )
    mode: str = Param(
        "max",
        description=(
            "Whether the monitored quantity improves when it falls, ``min``, "
            "or when it rises, ``max``."
        ),
    )
    patience: int = Param(
        5,
        ge=0,
        description=(
            "Number of validation checks without improvement after which "
            "training stops, 0 or greater. Checks during the model's warmup "
            "epochs are skipped."
        ),
    )


@register_class(key=SCORE_CHECKPOINT, component=CHECKPOINT)
class ScoreCheckpointConfig(Configuration):
    """Keep the epoch with the highest combined score."""

    monitor: str = Param(
        "val_score",
        description="Name of the logged quantity that the callback monitors.",
    )
    mode: str = Param(
        "max",
        description=(
            "Whether the monitored quantity improves when it falls, ``min``, "
            "or when it rises, ``max``."
        ),
    )


__all__ = [
    "GeneralizationLossScoreConfig",
    "LossCheckpointConfig",
    "LossEarlyStoppingConfig",
    "ScoreCheckpointConfig",
    "ScoreEarlyStoppingConfig",
]
