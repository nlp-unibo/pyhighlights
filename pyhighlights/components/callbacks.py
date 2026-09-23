"""Callbacks a run is monitored by, and the criterion that selects its epoch.

Early stopping and checkpointing used to be built inside
:class:`~pyhighlights.components.tasks.SPPTask` from one ``monitor`` string and
one ``patience``, with ``mode="min"`` written into both. That is three
decisions a study cannot make: it cannot maximise a metric, cannot monitor
anything but a single logged quantity, and cannot say that a model spends its
first epochs pretraining something the monitor knows nothing about. They are
registered components here, so a study configures them the way it configures
its losses and its metrics.

**One quantity decides both.** The callback that stops a run and the callback
that keeps an epoch have to agree, or a run reports a model its own stopping
rule did not choose: stop when two quantities are exhausted and the restored
epoch is whichever one the checkpoint happened to monitor.
:class:`GeneralizationLossScore` exists so that a study combining two
quantities still monitors one.
"""

from __future__ import annotations

from typing import Any, Mapping

import lightning as L
import torch as th
from lightning.pytorch.callbacks import Callback, EarlyStopping, ModelCheckpoint

#: The two Lightning internals :class:`WarmupEarlyStopping` and
#: :class:`WarmupModelCheckpoint` extend. They are private, and a release
#: renaming one would leave the subclass inheriting the base behaviour: no
#: error, no warning, and a pretraining epoch stopping and checkpointing a run
#: again. Checked here so that the upgrade fails at import instead.
_HOOKS = (
    (EarlyStopping, "_run_early_stopping_check"),
    (ModelCheckpoint, "_save_topk_checkpoint"),
)
for _base, _hook in _HOOKS:
    if not hasattr(_base, _hook):
        raise ImportError(
            f"lightning's {_base.__name__} no longer defines {_hook}, which "
            "pyhighlights extends to skip an epoch a model pretrains in. "
            "Port the warmup callbacks to the hook that replaced it"
        )

__all__ = [
    "GeneralizationLossScore",
    "MonitoredScore",
    "WarmupEarlyStopping",
    "WarmupModelCheckpoint",
    "warmup_epochs",
]


def warmup_epochs(module: L.LightningModule) -> int:
    """Training epochs before the monitored model starts learning.

    A model that pretrains a component over the first epochs of the training
    loop is not improving the thing a monitor watches, so a patience counted
    from epoch zero can exhaust before the model has taken a single step.
    G-RAT is that case -- it gates its optimizer on
    ``current_epoch >= pretrain_epochs`` -- and two of five seeds of the legal
    study's frozen arm stopped inside that window.

    The number comes from the model rather than from the task's configuration,
    because a task repeating it is a second place for it to be wrong. A model
    that pretrains outside the training loop, as DAR does in
    ``on_train_start``, costs no epochs and needs nothing here.
    """
    return int(getattr(module, "warmup_epochs", 0) or 0)


class MonitoredScore(Callback):
    """A callback that writes the quantity a run is monitored by.

    Marked as a class rather than recognised by name, so that
    :meth:`~pyhighlights.components.tasks.SPPTask.build_callbacks` can order
    every criterion before the callbacks that read one.
    """


class GeneralizationLossScore(MonitoredScore):
    """One monitored quantity out of a metric and a one-sided loss penalty.

    Maximises ``quality`` while charging for validation loss that has risen
    above its own best:

    .. math::

        score = quality - \\lambda \\cdot \\max(0, loss / loss_{opt} - 1)

    The penalty is Prechelt's **generalization loss** (Prechelt, 1998, *Early
    Stopping -- But When?*), with ``loss_opt`` the lowest validation loss seen
    so far. A *ratio* rather than a difference, and that matters: a weighted
    cross entropy at a class weight of 105 is unbounded while an F1 is not, so
    a difference would make ``coefficient`` a guess about scale. A relative
    regression is dimensionless, and the coefficient means one thing -- how
    much ``quality`` an epoch forfeits per unit of relative loss regression.

    Measured over the legal study's frozen arm, monitoring the rare class's F1
    alone accepted a 26 to 31% relative loss regression, and monitoring the
    loss alone gave up three to four points of that F1 at a regression under
    1%. The coefficient is what chooses between those, and it belongs to an
    architecture rather than to the library: a coefficient of 0 monitors
    ``quality`` alone and a large one monitors the loss alone, so both of the
    criteria it replaces are special cases of it.
    """

    def __init__(
        self,
        quality: str = "val_f1",
        loss: str = "val_loss",
        coefficient: float = 2.0,
        name: str = "val_score",
    ):
        super().__init__()
        if coefficient < 0:
            raise ValueError("coefficient must be non-negative")
        self.quality = quality
        self.loss = loss
        self.coefficient = coefficient
        self.name = name
        self.best_loss: float | None = None

    def state_dict(self) -> Mapping[str, Any]:
        """The floor the run is charged against, and only that.

        ``loss_opt`` is the whole memory of the criterion: a resumed run that
        forgot it would charge nothing for a regression it had already seen.
        The warmup window is not stored, because it is the model's. A run
        resumed against a model whose ``warmup_epochs`` has changed therefore
        keeps a floor set under the old window, which is a different model's
        loss.
        """
        return {"best_loss": self.best_loss}

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.best_loss = state_dict.get("best_loss")

    def on_validation_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        """Logged here, not in ``on_validation_epoch_end``.

        Lightning runs a callback's ``on_validation_epoch_end`` *before* the
        module's, so the metrics this reads are not logged yet at that point
        and the criterion would silently write nothing -- leaving early
        stopping to fail on a quantity that never appeared. ``on_validation_end``
        runs after the module has logged, and before ``EarlyStopping``'s own
        check on the same hook, provided this callback comes first in the
        list. :meth:`~pyhighlights.components.tasks.SPPTask.build_callbacks`
        puts it there rather than trusting the order it was given.
        """
        logged = trainer.callback_metrics
        if trainer.sanity_checking:
            # A validation epoch before training, over a model that has taken
            # no step. Not this criterion's business, and its metrics are not
            # evidence that the run logs what this reads.
            return
        missing = [name for name in (self.quality, self.loss) if name not in logged]
        if missing:
            # Returning quietly would leave `self.name` unwritten, and the
            # callbacks that monitor it skip a check they cannot make: the run
            # would train to `max_epochs` and be scored on its last epoch,
            # with nothing in `results.json` saying the criterion never ran.
            raise KeyError(
                f"{type(self).__name__} monitors {missing}, which nothing "
                f"logged. The run logged {sorted(logged)}"
            )
        if trainer.current_epoch < warmup_epochs(pl_module):
            # A pretraining epoch's loss is a different model's loss, so it
            # cannot set the floor the rest of the run is charged against.
            return

        quality = float(logged[self.quality])
        loss = float(logged[self.loss])
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
        if self.best_loss > 0:
            regression = max(0.0, loss / self.best_loss - 1.0)
        else:
            # A floor of zero has no ratio to take, and dividing by an epsilon
            # would charge a run that reached a validation loss of zero some
            # arbitrary multiple of 1e12. The absolute rise is the same number
            # when the floor is one, and it is bounded here.
            regression = max(0.0, loss)
        score = quality - self.coefficient * regression

        # `self.log` is refused on this hook, so the value is put where the
        # callbacks that read it look -- `trainer.callback_metrics` is what
        # both `EarlyStopping` and `ModelCheckpoint` consult -- and handed to
        # the logger separately so that it also lands in `metrics.csv` and a
        # learning curve can be drawn from it afterwards.
        trainer.callback_metrics[self.name] = th.tensor(score)
        if trainer.logger is not None:
            # `epoch` goes in the payload and not only in `step`. A
            # `LightningModule` logging with `on_epoch=True` adds that column
            # itself; a callback reaching the logger directly does not, and
            # `CSVLogger` then writes the score on a row whose `epoch` is
            # blank. Anything grouping `metrics.csv` by epoch -- which is what
            # drawing a learning curve is -- drops that row, so the quantity
            # the run was stopped and scored on is the one quantity missing
            # from its own curves.
            trainer.logger.log_metrics(
                {self.name: score, "epoch": trainer.current_epoch},
                step=trainer.current_epoch,
            )


class WarmupEarlyStopping(EarlyStopping):
    """Early stopping that starts counting when the model starts learning.

    Patience is about epochs the monitored model failed to improve on, and an
    epoch it did not train in is not one of those.
    """

    def _run_early_stopping_check(self, trainer: L.Trainer) -> None:
        if trainer.current_epoch < warmup_epochs(trainer.lightning_module):
            return
        super()._run_early_stopping_check(trainer)


class WarmupModelCheckpoint(ModelCheckpoint):
    """Checkpointing that ignores the epochs before the model trains.

    Without this the best epoch of a pretraining phase is a candidate for the
    epoch a run is scored on, and it holds a rationalizer that has taken no
    gradient step at all.
    """

    def _save_topk_checkpoint(
        self, trainer: L.Trainer, monitor_candidates: dict
    ) -> None:
        if trainer.current_epoch < warmup_epochs(trainer.lightning_module):
            return
        super()._save_topk_checkpoint(trainer, monitor_candidates)
