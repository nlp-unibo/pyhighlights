"""The GenSPP 2025 reproduction: its keys, its values, and its corpora."""

import zipfile
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
    # The paper's patience is the reproduction's own registration, not the
    # library's default of five.
    (stopping,) = [
        Registry.retrieve_configuration(registration_key=key)
        for key in task.callbacks
        if "early_stopping" in key.tags
    ]
    assert stopping.patience == 30
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


def test_the_toy_corpus_reads_the_published_artifact(tmp_path):
    # The Zenodo record holds the artifact, not a loose pickle, because the
    # artifact is what carries the manifest, the licence and the citation. A
    # local copy of it has to read the same as the published one.
    archive = tmp_path / "pyhighlights-genspp-toy-v1.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.write(toy_pickle(tmp_path), "toy_dataset.pkl")
        target.writestr("README.md", "# artifact")

    splits = GenSPPToyLoader(
        url=str(archive), sha256=None, directory=tmp_path / "cache"
    ).load()

    assert list(splits) == ["train", "val", "test"]
    assert sum(len(frame) for frame in splits.values()) == 10


def test_the_toy_corpus_defaults_to_the_published_artifact():
    loader = GenSPPToyLoader()

    # The version record rather than the concept one: the digest pins these
    # exact bytes, and a concept DOI resolves to whatever is newest.
    assert loader.url.endswith("pyhighlights-genspp-toy-v1.zip/content")
    assert "22711449" in loader.url
    assert loader.sha256 == (
        "5b0886163b215b932b242ce4910cd8d60b46fa79cfdfdde41e9646d99d9ebc92"
    )


def test_the_toy_corpus_refuses_when_it_is_given_nowhere_to_look():
    # A clear refusal beats silently synthesising a different corpus.
    with pytest.raises(ValueError, match="no download URL"):
        GenSPPToyLoader(url=None).load()


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
