"""Whether a selection holds the annotated evidence or a code for the label."""

import math

import pandas as pd
import pytest

from pyhighlights.components import codes
from pyhighlights.components.codes import (
    SelectionCodeAnalyzer,
    class_highlights,
    complete,
    complete_in_order,
    contains,
    gold_recall,
    runs,
    summarize,
)


def toy(rows, run="m", seed="1"):
    """Toy-like documents: one character per token, highlight at given positions."""
    return pd.DataFrame(
        [
            {
                "run": run,
                "seed": seed,
                "tokens": list(text),
                "selected": list(selected),
                "highlights": None if highlights is None else list(highlights),
                "label": label,
                "predicted": predicted,
            }
            for text, selected, highlights, label, predicted in rows
        ]
    )


def test_contains_finds_a_contiguous_subsequence_only():
    assert contains(list("xabcy"), list("abc"))
    assert not contains(list("xacby"), list("abc"))
    assert not contains(list("ab"), list("abc"))
    # An empty part is not evidence of anything.
    assert not contains(list("abc"), [])


def test_runs_split_selected_positions_at_gaps():
    assert runs([5, 1, 2, 7, 6]) == [[1, 2], [5, 6, 7]]
    assert runs([]) == []


def test_a_selection_is_complete_when_it_covers_every_annotated_position():
    tokens = list("xxabcxx")
    assert complete(tokens, [2, 3, 4], [2, 3, 4])
    assert complete(tokens, [0, 2, 3, 4], [2, 3, 4])
    # The lone `b` of a toy code is not the highlight.
    assert not complete(tokens, [3], [2, 3, 4])
    # A document without annotation has nothing to be complete against.
    assert not complete(tokens, [2, 3, 4], [])


def test_a_rival_selected_as_one_run_makes_a_selection_incomplete():
    """Spelling two classes' highlights at once is not evidence for one."""
    tokens = list("abaxabc")
    gold = [4, 5, 6]
    assert not complete(tokens, [0, 1, 2, 4, 5, 6], gold, rivals=[list("aba")])
    # The same characters with a gap do not spell the rival.
    assert complete(tokens, [0, 2, 4, 5, 6], gold, rivals=[list("aba")])


def test_complete_in_order_drops_positions():
    tokens = list("abxabc")
    # `a`, `b` early and `c` late read as `abc` once positions are dropped.
    assert complete_in_order(tokens, [0, 1, 5], [3, 4, 5])
    assert not complete(tokens, [0, 1, 5], [3, 4, 5])
    assert not complete_in_order(tokens, [0, 1], [3, 4, 5])
    assert not complete_in_order(tokens, [0, 1, 3, 4, 5], [3, 4, 5], [list("aba")])


def test_gold_recall_is_the_share_of_annotated_positions_selected():
    assert gold_recall([1, 2], [1, 2, 3, 4]) == pytest.approx(0.5)
    assert math.isnan(gold_recall([1], []))


def test_class_highlights_needs_exactly_one_highlight_per_class():
    frame = toy(
        [
            ("abax", [], [0, 1, 2], 0, 0),
            ("xbaa", [], [1, 2, 3], 1, 1),
            ("abcx", [], [0, 1, 2], 2, 2),
        ]
    )
    assert class_highlights(frame) == {
        0: ("a", "b", "a"),
        1: ("b", "a", "a"),
        2: ("a", "b", "c"),
    }

    several = toy([("abax", [], [0, 1, 2], 0, 0), ("abxc", [], [0, 1, 3], 0, 0)])
    with pytest.raises(ValueError, match="one highlight per class"):
        class_highlights(several)

    unannotated = toy([("abax", [], [0, 1, 2], 0, 0), ("xbaa", [], None, 1, 1)])
    with pytest.raises(ValueError, match=r"classes \[1\] have no annotated"):
        class_highlights(unannotated)


def test_summarize_reports_accuracy_and_completeness_per_run_and_seed():
    frame = toy(
        [
            # A code: correct, and the highlight is not selected.
            ("xabcx", [2], [1, 2, 3], 2, 2),
            # The whole highlight.
            ("xabcx", [1, 2, 3], [1, 2, 3], 2, 2),
            # The whole highlight, but also a rival as one run.
            ("abaabc", [0, 1, 2, 3, 4, 5], [3, 4, 5], 2, 2),
            # Class 0's single document.
            ("abaxx", [0, 1, 2], [0, 1, 2], 0, 1),
            # Unannotated: counts for accuracy only.
            ("xxxxx", [0], None, 0, 0),
        ]
    )
    plain = summarize(frame).iloc[0]
    assert plain["samples"] == 5
    assert plain["annotated"] == 4
    assert plain["accuracy"] == pytest.approx(4 / 5)
    assert plain["complete"] == pytest.approx(3 / 4)
    assert plain["gold_recall"] == pytest.approx((1 / 3 + 1 + 1 + 1) / 4)

    exclusive = summarize(frame, exclusive=True).iloc[0]
    assert exclusive["complete"] == pytest.approx(2 / 4)
    assert exclusive["complete_in_order"] == pytest.approx(2 / 4)


def test_codes_report_the_most_frequent_selection_of_each_class():
    frame = toy(
        [
            ("xabcx", [2], [1, 2, 3], 2, 2),
            ("abcxx", [1], [0, 1, 2], 2, 2),
            ("xxabc", [2, 3, 4], [2, 3, 4], 2, 2),
            ("abaxx", [1, 2], [0, 1, 2], 0, 0),
        ]
    )
    report = codes.codes(frame).set_index("label")
    assert report.loc[2, "code"] == ("b",)
    assert report.loc[2, "share"] == pytest.approx(2 / 3)
    assert report.loc[0, "size"] == 2
    assert summarize(frame).iloc[0]["code_size"] == 3


def test_the_analyzer_summarizes_what_the_prediction_analyzer_reads(monkeypatch):
    frame = toy(
        [("xabcx", [2], [1, 2, 3], 2, 2), ("abaxx", [0, 1, 2], [0, 1, 2], 0, 0)]
    )
    monkeypatch.setattr(codes.PredictionAnalyzer, "analyze", lambda self: frame.copy())
    analyzer = SelectionCodeAnalyzer("anywhere", exclusive=True)
    assert analyzer.analyze().iloc[0]["complete"] == pytest.approx(1 / 2)
    assert set(analyzer.codes()["label"]) == {0, 2}

    monkeypatch.setattr(
        codes.PredictionAnalyzer, "analyze", lambda self: pd.DataFrame()
    )
    assert analyzer.analyze().empty
    assert analyzer.codes().empty
