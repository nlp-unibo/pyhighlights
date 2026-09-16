"""What predicts a label without reading the evidence it is supposed to rest on."""

import pandas as pd
import pytest

from pyhighlights.components.loaders import ToyLoader
from pyhighlights.components.shortcuts import (
    ShortcutDetector,
    ablated,
    incidence,
    lengths,
    ngrams,
    scan,
)


def frame(rows):
    return pd.DataFrame(
        [
            {
                "sample_id": index,
                "text": "".join(tokens),
                "tokens": list(tokens),
                "label": label,
                "highlights": list(flags),
            }
            for index, (tokens, label, flags) in enumerate(rows)
        ]
    )


def test_an_ngram_is_counted_once_per_document_it_appears_in():
    """Presence, not count: whether the n-gram is there is the feature."""
    found = incidence([list("aab"), list("bbb")], max_length=2)

    assert found["a"] == [0]
    assert found["aa"] == [0]
    # `b` twice in document 0 and three times in document 1, one entry each.
    assert found["b"] == [0, 1]
    assert found["bb"] == [1]

    with pytest.raises(ValueError, match="max_length is at least 1"):
        incidence([list("ab")], max_length=0)


def test_a_feature_is_scored_by_the_best_rule_over_it():
    """Accuracy of "this class when present, that class when absent"."""
    # `x` is in both class-1 documents and neither class-0 one.
    report = scan({"x": [2, 3], "y": [0, 2]}, [0, 0, 1, 1])

    perfect = report[report["feature"] == "x"].iloc[0]
    assert perfect["accuracy"] == pytest.approx(1.0)
    assert perfect["support"] == 2
    assert perfect["baseline"] == pytest.approx(0.5)

    # `y` splits both classes evenly, so the best rule is the majority class.
    useless = report[report["feature"] == "y"].iloc[0]
    assert useless["accuracy"] == pytest.approx(0.5)
    assert useless["mutual_information"] == pytest.approx(0.0)

    # Ranked, so the shortcut is the row a reader sees first.
    assert report.iloc[0]["feature"] == "x"

    with pytest.raises(ValueError, match="nothing to scan"):
        scan({"x": []}, [])


def test_the_scan_finds_a_planted_ngram_and_the_control_does_not():
    """The permuted column is what a feature scores when the labels mean nothing."""
    rows = [(f"qq{'z' if label else 'w'}qq", label, [0] * 5) for label in [0, 1] * 30]
    report = ngrams(frame(rows), max_length=2)

    planted = report[report["feature"] == "z"].iloc[0]
    assert planted["accuracy"] == pytest.approx(1.0)
    # Chance cannot reach it, which is what makes the real score readable.
    assert planted["permuted"] < 0.8


def test_length_is_a_feature_no_ngram_scan_can_see():
    """A class whose documents are longer needs no content to be identified."""
    rows = [(("ab" * 4 if label else "ab" * 2), label, []) for label in [0, 1] * 30]
    rows = [(tokens, label, [0] * len(tokens)) for tokens, label, _ in rows]

    # Same characters, same n-grams, and only the length differs.
    assert ngrams(frame(rows), max_length=2)["accuracy"].max() < 0.99
    assert lengths(frame(rows))["accuracy"].max() == pytest.approx(1.0)


def test_ablation_removes_the_evidence_and_keeps_the_width():
    """The hole cannot spell anything, and it is as wide as what it replaced."""
    hole = ablated(frame([(list("abcd"), 0, [0, 1, 1, 0])]))
    row = hole.iloc[0]

    assert row["tokens"] == ["a", "▮", "▮", "d"]
    assert row["text"] == "a▮▮d"
    assert len(row["tokens"]) == 4


def test_the_shipped_toy_corpus_has_no_shortcut_left_once_the_highlight_is_gone():
    """The gate on the corpus the library actually registers.

    It has been wrong twice, and neither time was visible from the outside:
    inserting patterns made the *document* longer for the class with the longer
    pattern, and unequal pattern lengths made the *highlight* wider for it.
    Either one solves the task without reading a character.
    """
    loader = ToyLoader(sizes={"train": 600}, vocabulary_size=10, seed=1)
    frame = loader.load()["train"]

    assert len({len(row.tokens) for row in frame.itertuples()}) == 1
    assert len({sum(row.highlights) for row in frame.itertuples()}) == 1

    ShortcutDetector(max_length=4).check(frame)


def test_a_conjunction_corpus_survives_the_gate_and_its_patterns_still_inform():
    """A shared pattern is insufficient, which is not the same as uninformative.

    `baa` sits in classes 0 and 1, so its *absence* names class 2 exactly and it
    scores well above the baseline on its own. That is why the gate is the
    ablation and not the ranking: the ranking is full of features that are
    supposed to be there.
    """
    loader = ToyLoader(
        sizes={"train": 600},
        triggers=[["aba", "baa"], ["baa", "abb"], ["abb", "aba"]],
        vocabulary_size=10,
        seed=1,
    )
    frame = loader.load()["train"]
    detector = ShortcutDetector(max_length=4)

    report = detector.report(frame)
    shared = report[report["feature"] == "baa"].iloc[0]
    assert shared["accuracy"] > shared["baseline"] + 0.2
    # And no single pattern is enough to name the class.
    assert shared["accuracy"] < 1.0

    detector.check(frame)


def test_the_gate_names_the_shortcut_it_refuses():
    """Unequal pattern lengths make the highlight's width a codeword."""
    loader = ToyLoader(sizes={"train": 400}, triggers=("aa", "bcd"), seed=1)

    with pytest.raises(ValueError, match="predict the label with the highlight"):
        ShortcutDetector(max_length=4).check(loader.load()["train"])
