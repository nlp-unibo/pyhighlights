import json
import tarfile
from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.datasets import (
    ERASERLoader,
    HateXplainLoader,
    ToyLoader,
)
from pyhighlights.configurations.datasets import ERASER, HATEXPLAIN, TOY


def annotator(label: str, index: int) -> dict:
    return {"label": label, "annotator_id": index, "target": ["None"]}


def hatexplain(directory: Path) -> dict:
    """Posts covering a majority label, a tie, a normal post and a duplicate."""
    directory.mkdir(parents=True, exist_ok=True)
    posts = {
        "p1": {
            "post_id": "p1",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("offensive", 3),
            ],
            "rationales": [[1, 1, 0], [1, 0, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
        "p2": {
            "post_id": "p2",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("offensive", 2),
                annotator("normal", 3),
            ],
            "rationales": [[0, 1]],
            "post_tokens": ["who", "cares"],
        },
        "p3": {
            "post_id": "p3",
            "annotators": [annotator("normal", i) for i in (1, 2, 3)],
            "rationales": [],
            "post_tokens": ["nice", "day"],
        },
        # Same text as p1: id-based splits do not stop text leaking.
        "p4": {
            "post_id": "p4",
            "annotators": [
                annotator("hatespeech", 1),
                annotator("hatespeech", 2),
                annotator("normal", 3),
            ],
            "rationales": [[1, 1, 0], [1, 1, 0], [0, 1, 0]],
            "post_tokens": ["burn", "them", "all"],
        },
    }
    divisions = {"train": ["p1", "p2"], "val": ["p3"], "test": ["p4"]}
    (directory / "posts.json").write_text(json.dumps(posts))
    (directory / "divisions.json").write_text(json.dumps(divisions))
    return {
        "url": (directory / "posts.json").as_uri(),
        "divisions_url": (directory / "divisions.json").as_uri(),
        "directory": directory / "cache",
    }


def eraser(directory: Path) -> dict:
    """A miniature single-document ERASER task."""
    directory.mkdir(parents=True, exist_ok=True)
    documents = {"d1.txt": "a truly awful film\nnot worth it", "d2.txt": "a fine film"}
    rows = {
        "train.jsonl": [
            {
                "annotation_id": "d1.txt",
                "classification": "NEG",
                "docids": None,
                "evidences": [
                    [
                        {
                            "docid": "d1.txt",
                            "start_token": 1,
                            "end_token": 3,
                            "text": "truly awful",
                        }
                    ],
                    [
                        {
                            "docid": "d1.txt",
                            "start_token": 5,
                            "end_token": 7,
                            "text": "worth it",
                        }
                    ],
                ],
            }
        ],
        "val.jsonl": [
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": ["d2.txt"],
                "evidences": [],
            }
        ],
        "test.jsonl": [
            {
                "annotation_id": "d2.txt",
                "classification": "POS",
                "docids": None,
                "evidences": [
                    [
                        {
                            "docid": "d2.txt",
                            "start_token": 1,
                            "end_token": 2,
                            "text": "fine",
                        }
                    ]
                ],
            }
        ],
    }
    staging = directory / "movies"
    (staging / "docs").mkdir(parents=True)
    for name, text in documents.items():
        (staging / "docs" / name).write_text(text)
    for name, lines in rows.items():
        (staging / name).write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n"
        )

    archive = directory / "movies.tar.gz"
    with tarfile.open(archive, "w:gz") as target:
        target.add(staging, arcname="movies")
    return {"url": archive.as_uri(), "directory": directory / "cache"}


def test_toy_corpus_marks_its_trigger_and_repeats_for_a_seed():
    loader = ToyLoader(sizes={"train": 8, "test": 4}, length=6, seed=3)
    splits = loader.load()

    assert list(splits) == ["train", "test"]
    assert len(splits["train"]) == 8
    assert (loader.check_leakage()["ratio"] == 0).all()

    row = splits["train"].iloc[0]
    marked = [token for token, flag in zip(row.tokens, row.highlights) if flag]
    assert " ".join(marked) == "a great film"
    assert len(row.highlights) == len(row.tokens) == 9
    assert sorted(splits["train"]["label"].unique()) == [0, 1]

    assert ToyLoader(seed=3).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    assert (
        not ToyLoader(seed=4).load()["train"].equals(ToyLoader(seed=3).load()["train"])
    )

    with pytest.raises(ValueError, match="one trigger per class"):
        ToyLoader(triggers=["only one"])


def test_hatexplain_aggregates_labels_and_rationales(tmp_path):
    loader = HateXplainLoader(**hatexplain(tmp_path), remove_leakage=False)
    splits = loader.load()

    # p2 has no majority label and is dropped; p3 is normal, so nothing is
    # marked; p1 keeps the token both rationale vectors agree on.
    assert [len(frame) for frame in splits.values()] == [1, 1, 1]
    assert list(splits["train"]["text"]) == ["burn them all"]
    assert splits["train"]["highlights"].iloc[0] == [1, 0, 0]
    assert splits["val"]["highlights"].iloc[0] == [0, 0]
    assert splits["val"]["label"].iloc[0] == 1

    union = HateXplainLoader(
        **hatexplain(tmp_path), rationale="union", remove_leakage=False
    ).load()
    assert union["train"]["highlights"].iloc[0] == [1, 1, 0]
    intersection = HateXplainLoader(
        **hatexplain(tmp_path), rationale="intersection", remove_leakage=False
    ).load()
    assert intersection["train"]["highlights"].iloc[0] == [1, 0, 0]

    kept = HateXplainLoader(
        **hatexplain(tmp_path), ties="keep", remove_leakage=False
    ).load()
    assert len(kept["train"]) == 2

    with pytest.raises(ValueError, match="rationale must be"):
        HateXplainLoader(rationale="whatever")
    with pytest.raises(ValueError, match="ties must be"):
        HateXplainLoader(ties="whatever")


def test_hatexplain_splits_leak_by_text_and_are_repaired(tmp_path):
    settings = hatexplain(tmp_path)

    raw = HateXplainLoader(**settings, remove_leakage=False)
    report = raw.leakage().set_index(["left", "right"])
    assert report.loc[("train", "test"), "ratio"] == 1.0

    repaired = HateXplainLoader(**settings)
    assert (repaired.check_leakage()["ratio"] == 0).all()
    assert repaired.removed == {"train": 1, "val": 0, "test": 0}
    assert len(repaired.load()["train"]) == 0


def test_eraser_turns_evidence_spans_into_highlights(tmp_path):
    settings = eraser(tmp_path)
    splits = ERASERLoader(**settings, remove_leakage=False).load()

    train = splits["train"].iloc[0]
    assert train.tokens == ["a", "truly", "awful", "film", "not", "worth", "it"]
    assert train.highlights == [0, 1, 1, 0, 0, 1, 1]
    assert train.label == 0

    # An evidence-free row is annotated with nothing marked, not left unlabelled.
    assert splits["val"]["highlights"].iloc[0] == [0, 0, 0]
    assert splits["test"]["highlights"].iloc[0] == [0, 1, 0]
    assert splits["test"]["label"].iloc[0] == 1

    # val and test share a document, and the annotated split keeps it.
    repaired = ERASERLoader(**settings)
    assert (repaired.check_leakage()["ratio"] == 0).all()
    assert repaired.removed == {"train": 0, "val": 1, "test": 0}
    assert len(repaired.load()["test"]) == 1


def test_eraser_refuses_query_based_tasks_and_unknown_ones():
    with pytest.raises(ValueError, match="pairs a query with its document"):
        ERASERLoader(task="multirc")
    with pytest.raises(ValueError, match="task must be one of"):
        ERASERLoader(task="imdb")


def test_eraser_rejects_spans_outside_the_document(tmp_path):
    settings = eraser(tmp_path)
    root = tmp_path / "cache" / "eraser" / "movies" / "movies"
    loader = ERASERLoader(**settings)
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
        ERASERLoader(**settings).load()


def test_registered_loaders_build(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert isinstance(Registry.from_key(TOY, sizes={"train": 4}), ToyLoader)
    assert isinstance(
        Registry.from_key(HATEXPLAIN, **hatexplain(tmp_path / "hx")), HateXplainLoader
    )
    assert isinstance(
        Registry.from_key(ERASER, **eraser(tmp_path / "er")), ERASERLoader
    )
