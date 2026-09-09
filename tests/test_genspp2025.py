"""The GenSPP 2025 reproduction: its keys, its values, and its corpora."""

from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
import pyhighlights_benchmarks
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    HATEXPLAIN_GENSPP,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_PIPELINE,
    HATEXPLAIN_SPARSITY,
    HATEXPLAIN_SPARSITY_LOSS,
    SEEDS,
    TOY_BENCHMARK,
    TOY_FR_TASK,
    TOY_GENSPP,
    TOY_GENSPP_TRAINER,
    TOY_MGR,
)
from pyhighlights_benchmarks.genspp2025.corpora import GenSPPToyLoader
from tests.corpora import hatexplain


def build_registry():
    """The reproduction is what runs; the library is what it references."""
    return Registry.build(
        directory=Path(pyhighlights_benchmarks.__file__).parent,
        external_directories=[Path(pyhighlights.__file__).parent],
    )


def toy_pickle(directory: Path, rows: int = 10) -> Path:
    path = directory / "toy_dataset.pkl"
    pd.DataFrame(
        {
            "text": ["abcdefghij"[:4] + f"{index:06d}" for index in range(rows)],
            "label": [index % 3 for index in range(rows)],
            "structure_indexes": [[0, 1, 2] for _ in range(rows)],
        }
    ).to_pickle(path)
    return path


def test_both_corpora_register():
    valid, invalid = build_registry()

    paper = [key for key in valid if key.namespace == "genspp2025"]
    # Naming the keys: an invalid one is an experiment missing from the
    # benchmark rather than an error, so the failure has to say which.
    assert not invalid, sorted(str(key) for key in invalid)
    # Both halves, not whichever the filesystem yielded first: registering the
    # second script needs the re-entrant registration context of cinnamon 2.0.2.
    assert len([key for key in paper if "toy" in key.tags]) >= 16
    assert len([key for key in paper if "hatexplain" in key.tags]) >= 22


def test_the_released_hyperparameters_are_what_is_registered():
    build_registry()

    task = Registry.from_key(TOY_FR_TASK)
    assert task.seeds == SEEDS == [2023, 15451, 1337, 2001, 2080]
    assert task.batch_size == 64
    assert task.patience == 30
    assert task.trainer_args["max_epochs"] == 500

    # Three generators, and the baselines' hidden sizes: 8 for toy, 16 for
    # HateXplain, both over 25-dimensional embeddings.
    mgr = Registry.from_key(TOY_MGR)
    assert len(mgr.selectors) == 3
    assert mgr.selector_backbone.embedding.embedding_dim == 25
    assert mgr.selector_backbone.encoder.hidden_size == 8

    # GenSPP's own encoder is the genetic half's, not the baselines': one
    # direction, and 26 dimensions on toy against the baselines' 25 -- a row
    # per letter of the alphabet the corpus is built from.
    toy_genspp = Registry.from_key(TOY_GENSPP)
    assert toy_genspp.selector_backbone.embedding.embedding_dim == 26
    assert toy_genspp.selector_backbone.encoder.bidirectional is False
    hatexplain_genspp = Registry.from_key(HATEXPLAIN_GENSPP)
    assert hatexplain_genspp.selector_backbone.embedding.embedding_dim == 25
    assert hatexplain_genspp.selector_backbone.encoder.bidirectional is False

    grat = Registry.from_key(HATEXPLAIN_GRAT)
    assert grat.selector_backbone.encoder.hidden_size == 16
    assert grat.guide_decay == pytest.approx(1e-5)
    assert grat.pretrain_epochs == 10
    coefficients = {loss.name: loss.coefficient for loss in grat.losses}
    assert coefficients["guide"] == pytest.approx(2.5)
    assert coefficients["jsd"] == pytest.approx(1.5)
    assert "contiguity" not in coefficients

    # Sparsity is the one target that differs between the corpora.
    assert Registry.from_key(HATEXPLAIN_SPARSITY).threshold == pytest.approx(0.22)
    assert Registry.from_key(HATEXPLAIN_SPARSITY_LOSS).loss.threshold == pytest.approx(
        0.22
    )

    # So is the expected cross entropy the search calls a candidate good at.
    assert Registry.from_key(TOY_GENSPP_TRAINER).task_loss_limit == pytest.approx(0.1)
    assert Registry.from_key(
        HATEXPLAIN_GENSPP_TRAINER
    ).task_loss_limit == pytest.approx(0.6)

    benchmark = Registry.from_key(TOY_BENCHMARK)
    assert len(benchmark.tasks) == 5


def test_the_toy_corpus_is_read_as_characters(tmp_path):
    loader = GenSPPToyLoader(url=str(toy_pickle(tmp_path)))
    splits = loader.load()

    assert list(splits) == ["train", "val", "test"]
    row = splits["train"].iloc[0]
    # Tokens are characters, so the vocabulary is the alphabet.
    assert row["tokens"] == list(row["text"])
    assert sum(row["highlights"]) == 3
    assert row["highlights"][:3] == [1, 1, 1]

    # 80% train, a fifth of it held out, the rest test.
    assert len(splits["test"]) == 2
    assert len(splits["train"]) + len(splits["val"]) == 8

    # The validation draw is seeded, so two loads agree.
    again = GenSPPToyLoader(url=str(toy_pickle(tmp_path))).load()
    assert splits["val"]["text"].tolist() == again["val"]["text"].tolist()


def test_the_toy_corpus_says_it_has_nowhere_to_download_from():
    # The artifact is built but not yet published; a clear refusal beats
    # silently synthesising a different corpus.
    with pytest.raises(ValueError, match="no download URL yet"):
        GenSPPToyLoader().load()


def test_the_hatexplain_pipeline_folds_classes_before_it_counts_votes(tmp_path):
    build_registry()
    from pyhighlights.components.loaders import HateXplainLoader

    splits = HateXplainLoader(**hatexplain(tmp_path / "corpus")).load()
    pipeline = Registry.from_key(HATEXPLAIN_PIPELINE, expected_type=Preprocessor)
    processed = pipeline.process(splits)

    labels = dict(zip(processed["train"]["text"], processed["train"]["label"]))
    # "who cares" is annotated hatespeech, offensive and normal. Over three
    # classes that is a tie and the row is dropped; folding offensive into
    # hatespeech first makes it a two-to-one majority, which is the paper's.
    assert labels["who cares"] == 0
    assert set(processed["train"]["label"]) <= {0, 1}
    assert processed["val"]["label"].tolist() == [1]
