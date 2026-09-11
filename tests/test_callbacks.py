"""What monitors a run, and when it is allowed to start."""

from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

import pyhighlights
from pyhighlights.components.callbacks import (
    GeneralizationLossScore,
    WarmupEarlyStopping,
    WarmupModelCheckpoint,
    warmup_epochs,
)
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import (
    GENERALIZATION_LOSS_SCORE,
    GRU_FR,
    GRU_GRAT,
    LOSS_CHECKPOINT,
    LOSS_EARLY_STOPPING,
    SCORE_CHECKPOINT,
    SCORE_EARLY_STOPPING,
    TOY,
)


class Recorder(L.LightningModule):
    """A module that records what a callback logged on it."""

    warmup_epochs = 0

    def __init__(self):
        super().__init__()
        self.logged = {}

    def log(self, name, value, **kwargs):  # noqa: A003
        self.logged[name] = float(value)


class Fake:
    """Enough of a trainer for a callback that only reads metrics."""

    logger = None

    def __init__(self, metrics, epoch=0, sanity_checking=False):
        self.callback_metrics = {
            name: th.tensor(value) for name, value in metrics.items()
        }
        self.current_epoch = epoch
        self.sanity_checking = sanity_checking


def score(callback, module, **metrics):
    """The criterion's value, read where the callbacks that use it read it."""
    trainer = Fake(metrics)
    callback.on_validation_end(trainer, module)
    value = trainer.callback_metrics.get(callback.name)
    return None if value is None else float(value)


def test_the_score_charges_only_for_loss_above_its_own_best():
    """Prechelt's generalization loss: relative, and one-sided."""
    callback = GeneralizationLossScore(coefficient=2.0)
    module = Recorder()

    # First epoch sets the floor, so nothing is charged.
    assert score(callback, module, val_f1=0.30, val_loss=0.40) == pytest.approx(0.30)
    # A lower loss moves the floor down and is still not charged.
    assert score(callback, module, val_f1=0.35, val_loss=0.20) == pytest.approx(0.35)
    # A loss 50% above the best costs `coefficient` times that.
    assert score(callback, module, val_f1=0.50, val_loss=0.30) == pytest.approx(
        0.50 - 2.0 * 0.5
    )


def test_the_coefficient_spans_the_two_criteria_it_replaces():
    """0 monitors the metric alone; a large one monitors the loss alone."""
    metrics = [
        {"val_f1": 0.30, "val_loss": 0.10},  # sets the floor
        {"val_f1": 0.45, "val_loss": 0.13},  # +0.15 F1 for a 30% regression
    ]

    alone = [
        score(GeneralizationLossScore(coefficient=0.0), Recorder(), **m)
        for m in metrics
    ]
    # Without a penalty the second epoch wins, which is what accepted a 26 to
    # 31% regression on the legal study's frozen arm.
    assert alone[1] > alone[0]

    strict = GeneralizationLossScore(coefficient=5.0)
    module = Recorder()
    first = score(strict, module, **metrics[0])
    second = score(strict, module, **metrics[1])
    assert second < first


def test_a_pretraining_epoch_sets_no_floor_and_scores_nothing():
    """A guider's loss is a different model's loss."""
    callback = GeneralizationLossScore(coefficient=1.0)
    module = Recorder()
    module.warmup_epochs = 2

    warming = Fake({"val_f1": 0.9, "val_loss": 0.01}, epoch=0)
    callback.on_validation_end(warming, module)
    assert callback.name not in warming.callback_metrics
    assert callback.best_loss is None

    monitored = Fake({"val_f1": 0.2, "val_loss": 0.50}, epoch=2)
    callback.on_validation_end(monitored, module)
    # The floor is the first *monitored* epoch, not the pretraining minimum.
    assert callback.best_loss == pytest.approx(0.50)
    assert float(monitored.callback_metrics[callback.name]) == pytest.approx(0.2)


def test_sanity_checking_is_not_an_epoch():
    callback = GeneralizationLossScore()
    module = Recorder()
    callback.on_validation_end(
        Fake({"val_f1": 0.1, "val_loss": 9.0}, sanity_checking=True), module
    )
    assert callback.best_loss is None


def test_the_score_survives_a_resumed_run():
    callback = GeneralizationLossScore()
    module = Recorder()
    score(callback, module, val_f1=0.3, val_loss=0.2)

    resumed = GeneralizationLossScore()
    resumed.load_state_dict(callback.state_dict())

    # Without the floor a resumed run would charge nothing for a regression
    # it had already seen.
    assert resumed.best_loss == pytest.approx(0.2)


def test_warmup_epochs_comes_from_the_model():
    """G-RAT pretrains inside the training loop; nothing else does."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert warmup_epochs(Registry.from_key(GRU_GRAT)) == 10
    assert warmup_epochs(Registry.from_key(GRU_FR)) == 0


def test_early_stopping_does_not_count_a_pretraining_epoch(monkeypatch):
    stopping = WarmupEarlyStopping(monitor="val_loss", mode="min", patience=1)
    checked = []
    monkeypatch.setattr(
        EarlyStopping,
        "_run_early_stopping_check",
        lambda self, trainer: checked.append(trainer.current_epoch),
    )

    module = Recorder()
    module.warmup_epochs = 3
    for epoch in range(5):
        trainer = Fake({}, epoch=epoch)
        trainer.lightning_module = module
        stopping._run_early_stopping_check(trainer)

    assert checked == [3, 4]


def test_a_checkpoint_is_not_written_during_pretraining(monkeypatch):
    checkpoint = WarmupModelCheckpoint(monitor="val_loss", mode="min")
    saved = []
    monkeypatch.setattr(
        ModelCheckpoint,
        "_save_topk_checkpoint",
        lambda self, trainer, candidates: saved.append(trainer.current_epoch),
    )

    module = Recorder()
    module.warmup_epochs = 2
    for epoch in range(4):
        trainer = Fake({}, epoch=epoch)
        trainer.lightning_module = module
        checkpoint._save_topk_checkpoint(trainer, {})

    # An epoch the rationalizer sat out holds no rationalizer worth keeping.
    assert saved == [2, 3]


def test_a_negative_coefficient_is_refused():
    with pytest.raises(ValueError, match="coefficient"):
        GeneralizationLossScore(coefficient=-1.0)


def test_the_registered_pairs_monitor_one_quantity_each():
    """Stopping and selection have to agree, or a run reports a model its
    own stopping rule did not choose."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    for stopping_key, checkpoint_key in (
        (LOSS_EARLY_STOPPING, LOSS_CHECKPOINT),
        (SCORE_EARLY_STOPPING, SCORE_CHECKPOINT),
    ):
        stopping = Registry.from_key(stopping_key)
        checkpoint = Registry.from_key(checkpoint_key)
        assert stopping.monitor == checkpoint.monitor
        assert stopping.mode == checkpoint.mode

    # And the score pair monitors what the criterion writes.
    criterion = Registry.from_key(GENERALIZATION_LOSS_SCORE)
    assert Registry.from_key(SCORE_EARLY_STOPPING).monitor == criterion.name


def test_a_task_refuses_callbacks_that_monitor_different_quantities(tmp_path):
    """Stopping and selection disagreeing is not a configuration, it is a bug.

    Early stopping ends the run and the checkpoint chooses the epoch it is
    scored on. Give them two quantities and the reported model is not the one
    the stopping rule chose, and nothing downstream says so: `results.json`
    records the scores, not the argument behind them.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        name="mixed",
        save_path=str(tmp_path),
        callbacks=[LOSS_EARLY_STOPPING, SCORE_CHECKPOINT],
    )

    with pytest.raises(ValueError, match="monitoring different quantities"):
        task.build_callbacks(tmp_path)


def test_a_matching_pair_is_accepted(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        name="matched",
        save_path=str(tmp_path),
        callbacks=[LOSS_EARLY_STOPPING, LOSS_CHECKPOINT],
    )

    built = task.build_callbacks(tmp_path)

    assert len(built) == 2
    # And a criterion carries no `monitor`, so it never counts as a second one.
    scored = SPPTask(
        loader=TOY,
        model=GRU_FR,
        name="scored",
        save_path=str(tmp_path),
        callbacks=[GENERALIZATION_LOSS_SCORE, SCORE_EARLY_STOPPING, SCORE_CHECKPOINT],
    )
    assert len(scored.build_callbacks(tmp_path)) == 3
