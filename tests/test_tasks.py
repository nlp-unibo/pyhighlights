import json
from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.tasks import SPPTask, summarize, vocabulary
from pyhighlights.configurations.keys import (
    GRU_FR,
    LEAKAGE_REMOVER,
    TOY,
    TOY_TASK,
)
from pyhighlights.configurations.tasks import BINARY_METRICS


def build_registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def test_vocabulary_keeps_the_most_frequent_tokens_and_reserves_zero():
    frame = pd.DataFrame({"tokens": [["a", "b", "a"], ["a", "c"]]})

    # ``size`` is the width the embedding needs: three ids, one of them the
    # unknown token, so two tokens are mapped.
    assert vocabulary([frame], size=3) == {"a": 1, "b": 2}
    assert max(vocabulary([frame], size=3).values()) < 3

    with pytest.raises(ValueError, match="room for the unknown token"):
        vocabulary([frame], size=1)


def test_summarize_reports_the_spread_across_seeds():
    summary = summarize([{"accuracy": 0.5}, {"accuracy": 0.7}])

    assert summary["accuracy"]["mean"] == pytest.approx(0.6)
    assert summary["accuracy"]["std"] == pytest.approx(0.1)
    assert summary["accuracy"]["values"] == [0.5, 0.7]


def test_a_task_needs_seeds_and_a_positive_batch_size():
    with pytest.raises(ValueError, match="at least one seed"):
        SPPTask(loader=TOY, model=GRU_FR, seeds=[])
    with pytest.raises(ValueError, match="batch_size must be positive"):
        SPPTask(loader=TOY, model=GRU_FR, batch_size=0)


def test_a_task_trains_scores_and_writes_down_every_seed(tmp_path):
    build_registry()
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0, 1],
        store_predictions=True,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    results = task.run()

    assert isinstance(task, SPPTask)
    assert results["seeds"] == [0, 1]
    assert len(results["runs"]) == 2
    # Every metric the corpus asked for is reported on both splits.
    for split in ("val", "test"):
        for metric in (
            "accuracy",
            "f1",
            "highlight_f1",
            "highlight_iou",
            "selection_rate",
            "selection_size",
        ):
            assert f"{split}_{metric}" in results["summary"]

    accuracy = results["summary"]["test_accuracy"]
    assert accuracy["values"] == [run["test_accuracy"] for run in results["runs"]]
    assert accuracy["std"] >= 0.0

    written = json.loads((tmp_path / "toy" / "results.json").read_text())
    assert written == results
    assert (tmp_path / "toy" / "config.json").exists()

    for seed in (0, 1):
        directory = tmp_path / "toy" / f"seed={seed}"
        assert list(directory.glob("*.ckpt"))
        predictions = pd.read_pickle(directory / "predictions.pkl")
        assert {"highlight_mask", "class_logits", "y_true"} <= set(predictions[0])


def test_a_task_runs_without_a_validation_split(tmp_path):
    build_registry()
    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        test_metrics=BINARY_METRICS,
        name="no-val",
        save_path=str(tmp_path),
        batch_size=8,
        monitor="train_loss",
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    # A corpus split in two: nothing monitors validation, and the last epoch
    # is what gets scored.
    splits = task.splits()
    splits.pop("val")
    loaders = task.loaders(splits)

    results = task.fit(seed=0, loaders=loaders)
    assert not any(name.startswith("val_") for name in results)
    assert "test_accuracy" in results


def test_a_task_preprocesses_before_it_trains(tmp_path):
    """The preprocessor is a key like any other, and optional."""
    build_registry()
    plain = SPPTask(loader=TOY, model=GRU_FR, save_path=str(tmp_path))
    repaired = SPPTask(
        loader=TOY,
        model=GRU_FR,
        preprocessor=LEAKAGE_REMOVER,
        save_path=str(tmp_path),
    )

    # The toy corpus repeats its filler, so the repair has something to drop.
    assert sum(map(len, repaired.splits().values())) <= sum(
        map(len, plain.splits().values())
    )
    assert list(repaired.splits()) == list(plain.splits())
