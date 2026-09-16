import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.loaders import (
    R2A_SHA256,
    BeerLoader,
    ERASERLoader,
    HateXplainLoader,
    HotelLoader,
    MoviesLoader,
    R2ALoader,
    ToyLoader,
    to_examples,
)
from pyhighlights.configurations.keys import BEER, HATEXPLAIN, HOTEL, MOVIES, TOY
from tests.corpora import UNPINNED, eraser, hatexplain, r2a


def hotel(tmp_path: Path, **kwargs) -> HotelLoader:
    return HotelLoader(
        url=r2a(tmp_path), directory=tmp_path / "cache", **UNPINNED, **kwargs
    )


def test_r2a_downloads_once_and_parses_the_standard_columns(tmp_path):
    source = hotel(tmp_path)
    splits = source.load()

    assert set(splits) == {"train", "val", "test"}
    assert list(splits["train"].columns) == [
        "sample_id",
        "text",
        "tokens",
        "label",
        "highlights",
    ]
    # The splits come back as distributed: the annotated rows are still in
    # training, and nothing has been dropped.
    assert len(splits["train"]) == 3
    assert splits["train"]["highlights"].isna().all()

    test = splits["test"]
    assert test["tokens"].iloc[1] == ["close", "to", "the", "station"]
    assert test["highlights"].iloc[1] == [0, 0, 0, 1]
    assert list(test["sample_id"]) == [0, 1]

    # The archive is fetched once; a second load reuses the extracted copy.
    (source.directory / source.archive_name).unlink()
    assert source.read()["test"].equals(test)


def test_beer_and_hotel_accept_only_their_own_aspects(tmp_path):
    assert (
        BeerLoader(url=r2a(tmp_path), directory=tmp_path / "b", **UNPINNED).task
        == "beer0"
    )
    assert hotel(tmp_path).task == "hotel_Location"

    with pytest.raises(ValueError, match="task must be one of"):
        BeerLoader(task="hotel_Location")
    with pytest.raises(ValueError, match="task must be one of"):
        HotelLoader(task="beer0")
    # The shared parser still takes either.
    assert R2ALoader(task="beer1").task == "beer1"


def test_frames_convert_to_highlight_examples(tmp_path):
    datasets = hotel(tmp_path).datasets()
    example = datasets["test"][1]

    assert len(datasets["train"]) == 3
    assert example.highlights == (0, 0, 0, 1)
    assert datasets["train"][0].highlights is None


def test_misaligned_highlights_are_rejected(tmp_path):
    path = tmp_path / "broken.train"
    path.write_text(
        "task\tlabel\ttext\trationale\tpred_att\n"
        "hotel_Location\t1\ttwo tokens\t0 1 1\t0.1\n"
    )

    with pytest.raises(ValueError, match="misaligned"):
        R2ALoader.read_file(path)


def test_continuous_labels_are_rejected(tmp_path):
    path = tmp_path / "source.train"
    path.write_text("task\tlabel\ttext\nbeer0\t0.3\tno taste at all\n")

    with pytest.raises(ValueError, match="continuous labels"):
        R2ALoader.read_file(path)


def test_toy_corpus_marks_its_trigger_and_repeats_for_a_seed():
    loader = ToyLoader(sizes={"train": 8, "test": 4}, length=6, seed=3)
    splits = loader.load()

    assert list(splits) == ["train", "test"]
    assert len(splits["train"]) == 8

    row = splits["train"].iloc[0]
    marked = [token for token, flag in zip(row.tokens, row.highlights) if flag]
    # Tokens are characters, and the trigger is a character pattern.
    assert "".join(marked) == "aa"
    assert all(len(token) == 1 for token in row.tokens)
    assert row.text == "".join(row.tokens)
    # `length` is the document, patterns included: they overwrite filler rather
    # than lengthening it, so a longer pattern cannot make a longer document.
    assert len(row.highlights) == len(row.tokens) == 6
    assert sorted(splits["train"]["label"].unique()) == [0, 1]

    assert ToyLoader(seed=3).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    assert (
        not ToyLoader(seed=4).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    )

    with pytest.raises(ValueError, match="one trigger per class"):
        ToyLoader(triggers=["only one"])


def test_the_toy_filler_can_never_spell_a_trigger():
    """The one property the corpus has to have: the trigger is where it was put.

    The released generator reaches it by cleaning the sequence and rejecting a
    sample that satisfies another class; this reaches it by drawing filler from
    the letters no trigger uses.
    """
    loader = ToyLoader(sizes={"train": 200}, triggers=("aa", "bcd"), seed=11)
    frame = loader.load()["train"]

    assert not set(loader.alphabet) & set("abcd")
    for row in frame.itertuples():
        marked = "".join(
            token for token, flag in zip(row.tokens, row.highlights) if flag
        )
        assert (marked,) == loader.triggers[row.label]
        # Its own trigger appears once, and no other class's appears at all.
        assert row.text.count(marked) == 1
        assert loader.satisfied(row.text) == {row.label}

    with pytest.raises(ValueError, match="triggers leave"):
        ToyLoader(triggers=("abcdefghijklm", "nopqrstuvwxyz"), vocabulary_size=1)


def test_a_trigger_can_be_a_conjunction_no_single_pattern_identifies():
    """The corpus worth having: every pattern is shared by two classes.

    `baa` sits in classes 0 and 1, `abb` in 1 and 2, `aba` in 2 and 0, so a
    model that memorises one pattern cannot beat chance. Only the pair decides,
    which is the point of a conjunction.
    """
    triggers = [["aba", "baa"], ["baa", "abb"], ["abb", "aba"]]
    loader = ToyLoader(
        sizes={"train": 120}, triggers=triggers, vocabulary_size=10, seed=5
    )
    frame = loader.load()["train"]

    # No pattern belongs to one class.
    for pattern in ("aba", "baa", "abb"):
        holders = [label for label, t in enumerate(loader.triggers) if pattern in t]
        assert len(holders) == 2

    for row in frame.itertuples():
        # The sample satisfies its own class and no other. This is the property
        # the label rests on, so it is asserted rather than assumed.
        assert loader.satisfied(row.text) == {row.label}
        marked = [token for token, flag in zip(row.tokens, row.highlights) if flag]
        assert len(marked) == sum(len(p) for p in loader.triggers[row.label])


def test_a_conjunction_is_highlighted_as_separate_spans():
    """Two patterns are two spans, never one run: a filler character separates.

    Which is why the offsets are distinct. A contiguous highlight would make
    the corpus a worse test than the shape a real highlight has.
    """
    loader = ToyLoader(
        sizes={"train": 60},
        triggers=[["aba", "baa"], ["baa", "abb"], ["abb", "aba"]],
        vocabulary_size=10,
        seed=7,
    )

    for row in loader.load()["train"].itertuples():
        spans, previous = 0, 0
        for flag in row.highlights:
            spans += flag and not previous
            previous = flag
        assert spans == 2, (row.text, row.highlights)


def test_the_order_patterns_were_listed_in_is_not_a_feature():
    """Placement shuffles, so `aba` precedes `baa` about half the time.

    Listing order would otherwise be a channel: every class-0 sample would
    spell its first pattern first, and the position of either would say which
    class it is without reading it.
    """
    loader = ToyLoader(
        sizes={"train": 200},
        triggers=[["aba", "baa"], ["baa", "abb"], ["abb", "aba"]],
        vocabulary_size=10,
        seed=13,
    )
    frame = loader.load()["train"]
    first = [
        row.text.index("aba") < row.text.index("baa")
        for row in frame.itertuples()
        if row.label == 0
    ]

    assert 0.3 < sum(first) / len(first) < 0.7, sum(first) / len(first)


def test_a_class_covering_another_is_refused_rather_than_mislabelled():
    """Class 0 needs `aba`; class 1 needs `aba` and `baa`, so it holds both.

    Every class-1 draw satisfies class 0 as well, and no placement escapes it.
    A corpus whose labels cannot be read off its own rule is not one to write.
    """
    loader = ToyLoader(sizes={"train": 4}, triggers=["aba", ["aba", "baa"]], seed=0)

    with pytest.raises(ValueError, match="could not be placed"):
        loader.load()


def test_hatexplain_keeps_every_annotator_judgement(tmp_path):
    splits = HateXplainLoader(**hatexplain(tmp_path)).load()

    # Nothing is aggregated and nothing is dropped: the tie post survives, and
    # the label and highlights wait for a preprocessor.
    assert [len(frame) for frame in splits.values()] == [2, 1, 1]
    assert splits["train"]["label"].isna().all()
    assert splits["train"]["highlights"].isna().all()
    assert splits["train"]["annotator_labels"].iloc[0] == [
        "hatespeech",
        "hatespeech",
        "offensive",
    ]
    assert splits["train"]["annotator_highlights"].iloc[0] == [[1, 1, 0], [1, 0, 0]]


def test_unaggregated_splits_refuse_to_become_examples(tmp_path):
    loader = HateXplainLoader(**hatexplain(tmp_path))

    with pytest.raises(ValueError, match="AnnotationAggregator"):
        loader.datasets()


def test_eraser_turns_evidence_spans_into_highlights(tmp_path):
    splits = MoviesLoader(**eraser(tmp_path)).load()

    train = splits["train"].iloc[0]
    assert train.tokens == ["a", "truly", "awful", "film", "not", "worth", "it"]
    assert train.highlights == [0, 1, 1, 0, 0, 1, 1]
    assert train.label == 0

    # An evidence-free row is annotated with nothing marked, not left unlabelled.
    assert splits["val"]["highlights"].iloc[0] == [0, 0, 0]
    assert splits["test"]["highlights"].iloc[0] == [0, 1, 0]
    assert splits["test"]["label"].iloc[0] == 1


def test_eraser_refuses_query_based_tasks_and_unknown_ones():
    with pytest.raises(ValueError, match="pairs a query with its document"):
        ERASERLoader(task="multirc")
    with pytest.raises(ValueError, match="task must be one of"):
        ERASERLoader(task="imdb")


def test_eraser_rejects_spans_outside_the_document(tmp_path):
    settings = eraser(tmp_path)
    root = tmp_path / "cache" / "eraser" / "movies" / "movies"
    loader = MoviesLoader(**settings)
    loader.download()
    (root / "train.jsonl").write_text(
        json.dumps(
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": None,
                "evidences": [[{"docid": "d2.txt", "start_token": 2, "end_token": 9}]],
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="evidence span outside"):
        MoviesLoader(**settings).load()


def test_registered_loaders_build(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert isinstance(Registry.from_key(TOY, sizes={"train": 4}), ToyLoader)
    assert isinstance(
        Registry.from_key(HATEXPLAIN, **hatexplain(tmp_path / "hx")), HateXplainLoader
    )
    assert isinstance(
        Registry.from_key(MOVIES, **eraser(tmp_path / "er")), MoviesLoader
    )

    archive = r2a(tmp_path)
    beer = Registry.from_key(
        BEER, url=archive, directory=str(tmp_path / "b"), **UNPINNED
    )
    hotel_loader = Registry.from_key(
        HOTEL, url=archive, directory=str(tmp_path / "h"), **UNPINNED
    )
    assert isinstance(beer, BeerLoader) and beer.task == "beer0"
    assert isinstance(hotel_loader, HotelLoader)
    assert hotel_loader.task == "hotel_Location"


def test_registered_downloads_carry_the_digest_they_document(tmp_path):
    """The key is what a run builds, so the key is where the pin has to be.

    A constructor default is not enough: every registered run builds through
    the key, and a configuration naming `sha256=None` overrides the default
    silently. The documentation said these downloads were pinned while the
    registry said they were not.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    for key in (BEER, HOTEL):
        assert Registry.retrieve_configuration(key).sha256 == R2A_SHA256
    assert Registry.retrieve_configuration(MOVIES).sha256 == ERASERLoader.SHA256

    # HateXplain is two downloads, so it is two digests -- and two URLs at an
    # immutable commit rather than at a branch, since a digest pins bytes and
    # a branch name does not pin which bytes.
    hatexplain = Registry.retrieve_configuration(HATEXPLAIN)
    assert hatexplain.sha256 == HateXplainLoader.SHA256
    assert hatexplain.divisions_sha256 == HateXplainLoader.DIVISIONS_SHA256
    for url in (hatexplain.url, hatexplain.divisions_url):
        assert HateXplainLoader.COMMIT in url
        assert "/master/" not in url

    # And a stand-in archive opts out of the check explicitly, which is what
    # the loaders document.
    unpinned = Registry.from_key(
        BEER, url=r2a(tmp_path), directory=str(tmp_path / "b"), **UNPINNED
    )
    assert unpinned.sha256 is None


def test_to_examples_needs_the_standard_columns(tmp_path):
    frame = hotel(tmp_path).load()["train"].drop(columns=["label"])

    with pytest.raises(KeyError):
        to_examples(frame)


def test_a_generated_toy_corpus_saves_and_reads_back_identical(tmp_path):
    """The round trip that makes a second loader unnecessary.

    A toy corpus is generated, published and read back by whoever reproduces
    what it produced. If saving and reading are the same loader's two halves,
    a published corpus is a URL and a digest in a configuration; if they are
    not, it is a class somebody has to write for every dataset.
    """
    loader = ToyLoader(sizes={"train": 12, "val": 4, "test": 4}, seed=5)
    path = loader.save(tmp_path / "corpus.pkl")

    read = ToyLoader(url=str(path)).load()
    generated = loader.load()

    # The splits come back as they were generated, not re-cut by a ratio: the
    # file carries a `split` column for exactly this reason.
    assert list(read) == list(generated) == ["train", "val", "test"]
    for name, frame in generated.items():
        assert read[name]["text"].tolist() == frame["text"].tolist()
        assert read[name]["highlights"].tolist() == frame["highlights"].tolist()
        assert read[name]["label"].tolist() == frame["label"].tolist()


def test_a_flat_corpus_is_divided_by_the_ratios(tmp_path):
    """A corpus with no `split` column, which is what an older release is."""
    path = tmp_path / "flat.pkl"
    pd.DataFrame(
        {
            "sample_id": range(10),
            "text": [f"aa{index:04d}" for index in range(10)],
            "tokens": [list(f"aa{index:04d}") for index in range(10)],
            "label": [index % 2 for index in range(10)],
            "highlights": [[1, 1, 0, 0, 0, 0] for _ in range(10)],
        }
    ).to_pickle(path)

    splits = ToyLoader(url=str(path), train_ratio=0.8, val_ratio=0.2).load()

    # The GenSPP baselines' scheme: 80% train, a fifth of it held out, rest test.
    assert len(splits["test"]) == 2
    assert len(splits["train"]) + len(splits["val"]) == 8
    # Seeded, so two reads agree on which rows are validation.
    again = ToyLoader(url=str(path)).load()
    assert splits["val"]["text"].tolist() == again["val"]["text"].tolist()


def test_a_corpus_older_than_these_columns_is_converted_by_parse(tmp_path):
    """The hook a legacy schema overrides, rather than a loader of its own.

    The released GenSPP corpus stores `structure_indexes` -- the positions that
    are marked -- where this library stores a `highlights` vector. That is a
    difference in serialisation, not in what the corpus is, so it is fifteen
    lines of conversion rather than a second class.
    """
    path = tmp_path / "legacy.pkl"
    pd.DataFrame(
        {
            "text": [f"abc{index:03d}" for index in range(10)],
            "label": [index % 3 for index in range(10)],
            "structure_indexes": [[0, 1, 2] for _ in range(10)],
        }
    ).to_pickle(path)

    class LegacyToyLoader(ToyLoader):
        def parse(self, frame):
            frame = frame.copy()
            frame["highlights"] = [
                [1 if position in set(marked) else 0 for position in range(len(text))]
                for marked, text in zip(frame["structure_indexes"], frame["text"])
            ]
            return super().parse(frame)

    splits = LegacyToyLoader(url=str(path)).load()
    row = splits["train"].iloc[0]

    assert row["tokens"] == list(row["text"])
    assert row["highlights"][:3] == [1, 1, 1]
    assert sum(row["highlights"]) == 3

    # Unconverted, it says which column it could not find rather than guessing.
    with pytest.raises(ValueError, match="missing"):
        ToyLoader(url=str(path)).load()


def test_a_toy_corpus_reads_a_published_archive(tmp_path):
    """A record holds the archive, not a loose file: it carries the manifest."""
    inner = tmp_path / "corpus.pkl"
    ToyLoader(sizes={"train": 6, "test": 2}, seed=1).save(inner)
    archive = tmp_path / "toy-corpus.zip"
    with zipfile.ZipFile(archive, "w") as target:
        target.write(inner, "corpus.pkl")
        target.writestr("README.md", "# artifact")

    splits = ToyLoader(
        url=str(archive), sha256=None, directory=tmp_path / "cache"
    ).load()

    assert list(splits) == ["train", "test"]
    assert sum(len(frame) for frame in splits.values()) == 8


def test_a_configured_source_is_never_fallen_back_on(tmp_path):
    """The one thing the split into two loaders was protecting.

    A `url` that cannot be read has to raise. Generating instead would hand
    back a corpus of the right shape and different content, which is the
    failure nothing downstream can see -- every metric still computes.
    """
    loader = ToyLoader(url=str(tmp_path / "absent.pkl"))

    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        loader.load()

    with pytest.raises(ValueError, match="train_ratio"):
        ToyLoader(train_ratio=1.0)
    with pytest.raises(ValueError, match="val_ratio"):
        ToyLoader(val_ratio=1.0)
