"""What monitors a run, as configurations rather than as task parameters.

Two sets are registered. The ``loss`` pair reproduces what the task used to
build for itself -- early stopping and checkpointing on ``val_loss``,
minimised -- so a study that wants the old behaviour names it rather than
inheriting it silently. The ``score`` pair monitors
:class:`~pyhighlights.components.callbacks.GeneralizationLossScore`'s
combination and maximises it.

**Both callbacks of a pair monitor the same quantity**, and that is the point
of pairing them: the epoch that stops a run has to be the epoch the run is
scored on. A study is free to mix them, and gets a run whose reported model is
not the one its stopping rule chose.
"""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import NAMESPACE

EARLY_STOPPING = "pyhighlights.components.callbacks.WarmupEarlyStopping"
CHECKPOINT = "pyhighlights.components.callbacks.WarmupModelCheckpoint"


@register_class(
    name="callback",
    tags={"score", "generalization_loss"},
    namespace=NAMESPACE,
    component="pyhighlights.components.callbacks.GeneralizationLossScore",
)
class GeneralizationLossScoreConfig(Configuration):
    """A metric, charged for validation loss risen above its own best."""

    quality: str = Param("val_f1")
    loss: str = Param("val_loss")
    #: How much ``quality`` an epoch forfeits per unit of relative loss
    #: regression. Per architecture rather than per library: the loss curves
    #: differ, and 0 monitors the metric alone while a large value monitors
    #: the loss alone, so both are special cases of this.
    coefficient: float = Param(2.0, ge=0.0)
    name: str = Param("val_score")


@register_class(
    name="callback",
    tags={"early_stopping", "loss"},
    namespace=NAMESPACE,
    component=EARLY_STOPPING,
)
class LossEarlyStoppingConfig(Configuration):
    """Stop when the validation loss stops falling."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")
    patience: int = Param(5, ge=0)


@register_class(
    name="callback",
    tags={"checkpoint", "loss"},
    namespace=NAMESPACE,
    component=CHECKPOINT,
)
class LossCheckpointConfig(Configuration):
    """Keep the epoch with the lowest validation loss."""

    monitor: str = Param("val_loss")
    mode: str = Param("min")


@register_class(
    name="callback",
    tags={"early_stopping", "score"},
    namespace=NAMESPACE,
    component=EARLY_STOPPING,
)
class ScoreEarlyStoppingConfig(Configuration):
    """Stop when the combined score stops rising."""

    monitor: str = Param("val_score")
    mode: str = Param("max")
    patience: int = Param(5, ge=0)


@register_class(
    name="callback",
    tags={"checkpoint", "score"},
    namespace=NAMESPACE,
    component=CHECKPOINT,
)
class ScoreCheckpointConfig(Configuration):
    """Keep the epoch with the highest combined score."""

    monitor: str = Param("val_score")
    mode: str = Param("max")


__all__ = [
    "GeneralizationLossScoreConfig",
    "LossCheckpointConfig",
    "LossEarlyStoppingConfig",
    "ScoreCheckpointConfig",
    "ScoreEarlyStoppingConfig",
]
