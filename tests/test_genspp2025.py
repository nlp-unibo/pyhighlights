"""The GenSPP 2025 reproduction: its keys, its values, and its corpora."""

import zipfile
from pathlib import Path

import pandas as pd
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
import pyhighlights_benchmarks
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights_benchmarks.genspp2025.configurations.hatexplain.keys import (
    HATEXPLAIN_FR_TASK,
    HATEXPLAIN_GENSPP,
    HATEXPLAIN_GENSPP_TASK,
    HATEXPLAIN_GENSPP_TRAINER,
    HATEXPLAIN_GRAT,
    HATEXPLAIN_PIPELINE,
    HATEXPLAIN_SPARSITY,
    HATEXPLAIN_SPARSITY_LOSS,
)
from pyhighlights_benchmarks.genspp2025.configurations.keys import SEEDS
from pyhighlights_benchmarks.genspp2025.configurations.toy.keys import (
    TOY,
    TOY_BENCHMARK,
    TOY_FR_TASK,
    TOY_GENSPP,
    TOY_GENSPP_TASK,
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
    # direction, and 26 dimensions on toy against the baselines' 25. Both are
    # one-hot widths rather than projection sizes: the corpus uses twenty-four
    # characters, so the baselines' 25 is the vocabulary with its padding id
    # and the genetic half's 26 leaves two columns always zero.
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


def test_the_two_halves_of_hatexplain_build_their_vocabularies_differently(tmp_path):
    """The release does not embed HateXplain the same way twice.

    ``baselines/configurations/model.py`` sets ``use_pretrained_only=True``, and
    under that flag ``GloVeEmbedderCollator.fit`` discards the dataframe it is
    handed and takes all of ``twitter.27B`` as its vocabulary -- so no
    evaluation token the file covers is ever unknown.

    The genetic half does the opposite: ``Dataset.__build_tokenizer`` reads the
    training split only, and ``__embed_texts`` resolves an unknown id through a
    detokenizer built from it, so it embeds as zeros.

    Measured on these splits, the difference reaches 5.4% of validation tokens
    and 5.6% of test tokens.
    """
    build_registry()
    vectors = tmp_path / "vectors.txt"
    vectors.write_text(f"the {' '.join(['0.1'] * 25)}\n")

    baseline = Registry.from_key(
        HATEXPLAIN_FR_TASK, save_path=str(tmp_path), embeddings=str(vectors)
    )
    assert baseline.vocabulary_from == "vectors"

    genetic = Registry.from_key(
        HATEXPLAIN_GENSPP_TASK, save_path=str(tmp_path), embeddings=str(vectors)
    )
    assert genetic.vocabulary_from == "corpus"

    # Both refuse to be built without the file. The configuration used to
    # declare a vocabulary of two, so forgetting it trained on {'the': 1}.
    for key in (HATEXPLAIN_FR_TASK, HATEXPLAIN_GENSPP_TASK):
        with pytest.raises(ValueError, match="given none"):
            Registry.from_key(key, save_path=str(tmp_path))


def test_the_toy_corpus_reaches_a_model_as_one_hot(tmp_path):
    """The release does not embed this corpus, it one-hots it.

    Both halves build a one-hot matrix and hand it over -- the baselines
    through ``OneHotEmbedderCollator``, the genetic half through
    ``OneHotEmbedder``. This reproduction froze a *random* table instead,
    which is a different corpus to learn from: its rows had norm 5.16 and
    reached a cosine of 0.58 with one another, where one-hot rows are
    orthonormal and the padding row is zero.
    """
    build_registry()
    splits = GenSPPToyLoader(url=str(toy_pickle(tmp_path))).load()

    for key, width in ((TOY_FR_TASK, 25), (TOY_GENSPP_TASK, 26)):
        task = Registry.from_key(key, save_path=str(tmp_path))
        assert task.one_hot_embeddings == width
        task.tokenizer(splits)
        matrix = task._embedding_matrix

        assert matrix.shape == (25, width)
        # Row zero is the unknown and padding id, and contributes nothing.
        assert not matrix[0].any()
        # Every other row is a distinct basis vector.
        assert th.equal(matrix[1:].sum(dim=1), th.ones(24))
        assert th.equal(matrix[1:] @ matrix[1:].T, th.eye(24))

    # And the table the baselines' model is built with is that matrix, rather
    # than the random one its `vocab_size` sized.
    task = Registry.from_key(TOY_FR_TASK, save_path=str(tmp_path))
    task.tokenizer(splits)
    table = task.build_model().selector_backbone.embedding.weight
    assert th.equal(table, task._embedding_matrix)
    assert table.requires_grad is False


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


def test_the_registered_toy_key_names_the_published_artifact():
    """The documented reproduction key has to build without a private override.

    Registered with `url=None`, it did not: `Registry.from_key(TOY)` returned
    a loader with nowhere to read the corpus from, and `load()` refused.
    """
    build_registry()
    loader = Registry.from_key(TOY, expected_type=GenSPPToyLoader)

    assert loader.url == GenSPPToyLoader.URL
    assert loader.sha256 == GenSPPToyLoader.SHA256


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
