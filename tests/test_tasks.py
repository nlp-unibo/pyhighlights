import json
from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.tasks import (
    ClassWeightsTask,
    GenSPPTask,
    SPPTask,
    summarize,
    vocabulary,
)
from pyhighlights.configurations.keys import (
    CLASS_WEIGHTS,
    GRU_FR,
    GRU_GENSPP,
    LEAKAGE_REMOVER,
    TOY,
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
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

    # One directory per run, named for the moment it started.
    (run,) = (tmp_path / "toy").iterdir()
    written = json.loads((run / "results.json").read_text())
    assert written == results
    assert (run / "manifest.json").exists()

    for seed in (0, 1):
        assert list((run / f"seed={seed}").glob("*.ckpt"))
        # Beside the run, not inside the checkpoint directory: a reader that
        # finds them there can say which run they belong to.
        predictions = pd.read_pickle(run / f"predictions-seed={seed}.pkl")
        assert {"highlight_mask", "class_logits", "y_true", "word_ids"} <= set(
            predictions[0]
        )


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


def test_a_genspp_task_searches_scores_and_writes_down_its_generations(tmp_path):
    build_registry()
    task = Registry.from_key(
        TOY_GENSPP_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        store_predictions=True,
    )
    assert isinstance(task, GenSPPTask)
    # The model is the search's, not a second key that could disagree with it.
    assert task.search == TOY_GENSPP_TRAINER
    assert task.model == GRU_GENSPP

    results = task.run()

    for metric in ("accuracy", "f1", "highlight_f1", "selection_rate"):
        assert f"test_{metric}" in results["summary"]
        assert f"val_{metric}" in results["summary"]
    # Nothing trains the winner, so no split named ``train`` is ever scored.
    assert not any(name.startswith("train_") for name in results["summary"])

    (run,) = (tmp_path / "toy-genspp").iterdir()
    directory = run / "seed=0"
    assert (directory / "best.ckpt").exists()
    predictions = pd.read_pickle(run / "predictions-seed=0.pkl")
    assert {"highlight_mask", "class_logits", "y_true", "word_ids"} <= set(
        predictions[0]
    )

    # One entry per generation: the best fitness the search reached in it.
    progress = json.loads((directory / "search.json").read_text())
    assert len(progress["training_progress"]) == 1


def test_a_genspp_task_needs_a_validation_split(tmp_path):
    build_registry()
    task = Registry.from_key(TOY_GENSPP_TASK, save_path=str(tmp_path))
    splits = task.splits()
    splits.pop("val")

    # Fitness is what the search selects on, and it is measured on validation.
    with pytest.raises(ValueError, match="validation split"):
        task.fit(seed=0, loaders=task.loaders(splits))


def test_a_supervised_task_trains_against_the_annotation(tmp_path):
    build_registry()
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        highlight_supervision=True,
        highlight_coefficient=0.5,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    model = task.build_model()
    assert [loss.name for loss in model.losses][-1] == "highlight"
    assert model.losses[-1].coefficient == 0.5

    results = task.run()
    # The supervised term is reported like any other, so a table can show what
    # the ceiling cost in classification loss.
    assert "val_highlight" in results["summary"]
    assert "test_highlight" in results["summary"]


def test_supervision_is_refused_when_the_train_split_carries_no_annotation(tmp_path):
    build_registry()
    task = Registry.from_key(
        TOY_TASK, save_path=str(tmp_path), highlight_supervision=True
    )
    # The toy corpus annotates every split, so this one passes.
    task.check_supervision(task.splits())

    unannotated = pd.DataFrame({"highlights": [None, None]})
    # Unannotated positions are padded with -1 and skipped by the criterion, so
    # a corpus annotated on test alone would train as an unsupervised run does.
    with pytest.raises(ValueError, match="annotated train split"):
        task.check_supervision({"train": unannotated})
    with pytest.raises(ValueError, match="annotated train split"):
        task.check_supervision({"test": unannotated})


def test_genspp_refuses_highlight_supervision(tmp_path):
    build_registry()
    # No gradient reaches the generator: the loss would be built and never
    # train anything.
    with pytest.raises(ValueError, match="nothing to guide"):
        Registry.from_key(
            TOY_GENSPP_TASK, save_path=str(tmp_path), highlight_supervision=True
        )


def test_a_task_can_embed_its_tokens_with_a_vector_file(tmp_path):
    build_registry()
    # Two of the toy corpus's own tokens, so the vocabulary covers something.
    vectors = tmp_path / "vectors.txt"
    # The file's width has to be the backbone's embedding_dim; GRU_FR uses 128.
    vectors.write_text(
        "".join(
            f"{token} {' '.join(['0.1'] * 128)}\n" for token in ("a", "great", "film")
        )
    )

    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        save_path=str(tmp_path),
        batch_size=8,
        embeddings=str(vectors),
    )
    tokenizer = task.tokenizer(task.splits())
    assert set(tokenizer.vocabulary) <= {"a", "great", "film"}

    model = task.build_model()
    # The table is sized to the file, not to vocabulary_size, and the ids the
    # tokenizer hands out index it.
    assert (
        model.selector_backbone.embedding.num_embeddings
        == len(tokenizer.vocabulary) + 1
    )
    assert model.selector_backbone.embedding.embedding_dim == 128

    written = json.loads((task.serialize({"runs": []}) / "manifest.json").read_text())
    # A matrix is not a setting, so it stays out of what a run says it was.
    assert "embedding_matrix" not in written["settings"]


def test_a_task_embeds_its_tokens_one_way_or_the_other(tmp_path):
    with pytest.raises(ValueError, match="not both"):
        SPPTask(
            loader=TOY,
            model=GRU_FR,
            embeddings=str(tmp_path / "vectors.txt"),
            pretrained_model_card="distilbert-base-uncased",
        )


def test_a_class_weights_task_writes_down_what_it_read(tmp_path):
    """The whole result is the weights, and the counts they came from."""
    build_registry()
    task = ClassWeightsTask(loader=TOY, weights=CLASS_WEIGHTS, save_path=str(tmp_path))
    results = task.run()

    # The toy corpus alternates its two classes, so the weights are both 1.
    assert results["weights"] == [1.0, 1.0]
    assert results["counts"] == {"0": 32, "1": 32}
    assert results["rows"] == {"train": 64, "val": 16, "test": 16}

    written = json.loads((task.directory / "results.json").read_text())
    assert written == results

    # The same manifest every other task writes: which corpus, and which key.
    manifest = json.loads((task.directory / "manifest.json").read_text())
    assert manifest["settings"]["loader"]["key"] == str(TOY)
    assert manifest["settings"]["weights"]["split"] == "train"


def test_a_class_weights_task_records_the_preprocessing_it_weighed_after(tmp_path):
    """Frequencies are what preprocessing leaves, so the manifest names it."""
    build_registry()
    task = ClassWeightsTask(
        loader=TOY,
        weights=CLASS_WEIGHTS,
        preprocessor=LEAKAGE_REMOVER,
        save_path=str(tmp_path),
    )
    task.run()

    manifest = json.loads((task.directory / "manifest.json").read_text())
    assert manifest["settings"]["preprocessor"]["priority"] == ["test", "val", "train"]
