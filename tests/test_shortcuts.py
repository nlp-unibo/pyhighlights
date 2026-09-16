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


def test_contamination_is_what_stops_a_fragment_from_classifying():
    """`bc` occurs nowhere but inside `abc` until chunks are scattered about.

    This is contamination's acceptance test, and the reason it is worth having
    the scan: the released quality check reported that a truncated selection
    scores badly against the *annotation*, which was never in doubt. What
    matters is that it scores badly at the *task*, and only contamination makes
    it do so.
    """
    detector = ShortcutDetector(max_length=4)
    scores = []
    for contaminations in (0, 4):
        loader = ToyLoader(
            sizes={"train": 900},
            triggers=("aba", "baa", "abc"),
            vocabulary_size=10,
            contaminations=contaminations,
            seed=1,
        )
        corpus = loader.load()["train"]
        report = detector.report(corpus).set_index("feature")
        scores.append((report.loc["abc", "accuracy"], report.loc["bc", "accuracy"]))
        detector.check(corpus)

    (gold_clean, partial_clean), (gold_dirty, partial_dirty) = scores
    # Uncontaminated, the fragment is worth exactly as much as the pattern.
    assert partial_clean == pytest.approx(gold_clean)
    # Contaminated, the pattern is untouched and the fragment has lost a lot.
    assert gold_dirty == pytest.approx(gold_clean)
    assert partial_dirty < partial_clean - 0.1


def test_contamination_leaves_the_highlight_and_the_length_alone():
    """Chunks go in the filler: the annotation still marks exactly the patterns."""
    loader = ToyLoader(
        sizes={"train": 200},
        triggers=("aba", "baa", "abc"),
        vocabulary_size=10,
        contaminations=4,
        seed=2,
    )

    for row in loader.load()["train"].itertuples():
        assert len(row.tokens) == 20
        marked = "".join(t for t, flag in zip(row.tokens, row.highlights) if flag)
        assert marked == loader.triggers[row.label][0]
        # Exactly once, so the annotation is the whole truth about the pattern.
        assert row.text.count(marked) == 1
        assert loader.satisfied(row.text) == {row.label}


def test_a_chunk_is_a_proper_piece_that_spells_no_pattern():
    loader = ToyLoader(
        triggers=("aba", "baa", "abc"), vocabulary_size=10, contaminations=1
    )
    patterns = {p for trigger in loader.triggers for p in trigger}

    for chunk in loader.chunks:
        assert any(chunk in pattern and chunk != pattern for pattern in patterns)
        assert loader.satisfied(chunk) == set()

    # Patterns of two characters leave nothing proper to cut at min_chunk=2.
    with pytest.raises(ValueError, match="nothing to contaminate with"):
        ToyLoader(triggers=("aa", "bc"), contaminations=1)

    with pytest.raises(ValueError, match="contaminations and the gaps"):
        ToyLoader(triggers=("aba", "baa"), length=8, contaminations=4)


def test_ablation_rejoins_text_the_way_the_corpus_spells_it():
    """A word corpus is not a character one: `"".join` glues its words together.

    The scan reads `tokens` and never `text`, so nothing was scored wrongly,
    but an ablated word corpus reading `the▮brownfox` is one nobody can
    check by eye.
    """
    words = frame([(["the", "quick", "brown", "fox"], 0, [0, 1, 0, 0])])

    assert ablated(words, separator=" ").iloc[0]["text"] == "the ▮ brown fox"
    # Characters are the default, and join with nothing.
    assert ablated(words).iloc[0]["text"] == "the▮brownfox"

    # And the detector passes its own separator down, so `check` is consistent
    # with the `ngrams` call beside it.
    detector = ShortcutDetector(separator=" ")
    assert detector.separator == " "
