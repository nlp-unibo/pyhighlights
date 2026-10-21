"""Whether a selection holds the annotated evidence or a code for the label."""

import math
import sys

import numpy as np
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
    kept_by_label,
    mask_features,
    mask_probes,
    runs,
    selection_entropy,
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


def test_summarize_scores_positions_and_splits_accuracy_by_what_was_selected():
    frame = toy(
        [
            ("xabcx", [1, 2, 3], [1, 2, 3], 1, 0),  # complete, wrong
            ("xabcx", [2], [1, 2, 3], 1, 1),  # partial, right
            ("xabcx", [0, 4], [1, 2, 3], 1, 1),  # missed, right
            ("xabcx", [], [1, 2, 3], 1, 0),  # empty, wrong
            # Unannotated: its selection is no false positive.
            ("xxxxx", [0, 1, 2, 3, 4], None, 0, 0),
        ]
    )
    row = summarize(frame).iloc[0]
    assert row["empty"] == pytest.approx(1 / 5)
    assert row["highlight_precision"] == pytest.approx(4 / 6)
    assert row["highlight_recall"] == pytest.approx(4 / 12)
    assert row["highlight_f1"] == pytest.approx(8 / 18)
    assert row["missed"] == pytest.approx(1 / 4)
    assert row["accuracy_complete"] == 0
    assert row["accuracy_partial"] == 1
    assert row["accuracy_missed"] == 1


def test_summarize_reports_nan_where_nothing_is_annotated():
    row = summarize(toy([("xxxxx", [0], None, 0, 0)])).iloc[0]
    for column in ("highlight_precision", "highlight_f1", "accuracy_missed"):
        assert math.isnan(row[column])


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


def coded(documents=200, seed=0):
    """A model that decides from how much it keeps, not from what."""
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(documents):
        label = index % 2
        length = int(rng.integers(10, 30))
        kept = 1 if label else 6
        selected = sorted(rng.choice(length, size=kept, replace=False).tolist())
        rows.append(("x" * length, selected, None, label, label))
    return toy(rows)


def test_mask_features_describe_shape_without_text():
    features = mask_features(list("abcdefghij"), [0, 9])
    assert features[:3] == [10, 2, pytest.approx(0.2)]
    histogram = features[3:]
    assert len(histogram) == codes.BINS
    assert sum(histogram) == pytest.approx(1.0)
    assert histogram[0] == pytest.approx(0.5) and histogram[-1] == pytest.approx(0.5)


def test_a_count_code_is_found_by_the_count_channel_and_not_by_its_control():
    report = mask_probes(coded()).iloc[0]
    assert report["documents"] == 200
    assert report["f1"] > 0.95
    assert report["channel_count"] > 0.95
    assert report["permuted_f1"] < 0.75
    assert report["mcc"] > 0.9


def test_a_run_with_one_decision_reports_no_probe_of_its_decisions():
    """A model that always predicts one class has nothing to reproduce."""
    frame = coded().assign(predicted=0)
    report = mask_probes(frame).iloc[0]
    for column in ("f1", "permuted_f1", "mcc", "channel_count"):
        assert math.isnan(report[column])
    assert report["on_label"] > 0.95

    # A class with fewer members than folds is as unprobeable as no class.
    frame.loc[0, "predicted"] = 1
    assert math.isnan(mask_probes(frame).iloc[0]["f1"])


def test_short_documents_are_left_out_of_the_probes():
    frame = pd.concat([coded(), toy([("xyz", [0], None, 0, 0)])], ignore_index=True)
    assert mask_probes(frame).iloc[0]["documents"] == 200


def test_kept_by_label_exposes_a_count_code():
    report = kept_by_label(coded()).set_index("label")
    assert report.loc[0, "kept"] == pytest.approx(6)
    assert report.loc[1, "kept"] == pytest.approx(1)


def test_the_probes_name_the_missing_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sklearn.metrics", None)
    with pytest.raises(ImportError, match=r"pyhighlights\[analysis\]"):
        mask_probes(coded())


def test_the_analyzer_reports_probes_and_kept(monkeypatch):
    frame = coded()
    monkeypatch.setattr(codes.PredictionAnalyzer, "analyze", lambda self: frame.copy())
    analyzer = SelectionCodeAnalyzer("anywhere")
    assert analyzer.probes().iloc[0]["f1"] > 0.95
    assert len(analyzer.kept()) == 2

    monkeypatch.setattr(
        codes.PredictionAnalyzer, "analyze", lambda self: pd.DataFrame()
    )
    assert analyzer.probes().empty
    assert analyzer.kept().empty


def test_a_class_that_always_selects_one_token_has_zero_entropy():
    frame = toy([("abcdefgh", [1], None, 0, 0), ("hgfedcba", [6], None, 0, 0)])
    report = selection_entropy(frame).iloc[0]
    assert report["kept"] == 2
    assert report["entropy"] == pytest.approx(0.0)
    assert report["random_entropy"] > 0.0
    assert math.isnan(report["gold_entropy"])


def test_gold_entropy_is_the_reference_for_evidence_with_few_tokens():
    """A toy highlight such as `aba` repeats tokens, so it scores low as well."""
    frame = toy(
        [("xabax", [1, 2, 3], [1, 2, 3], 0, 0), ("abaxx", [1], [0, 1, 2], 0, 0)]
    )
    report = selection_entropy(frame).iloc[0]
    # Four `a` and two `b` annotated: the entropy of `aba` itself.
    assert report["gold_entropy"] == pytest.approx(0.918, abs=1e-3)


def test_random_selections_score_near_their_reference():
    rng = np.random.default_rng(1)
    alphabet = list("abcdefghijklmnop")
    rows = []
    for _ in range(300):
        tokens = list(rng.choice(alphabet, size=20))
        selected = rng.choice(20, size=5, replace=False).tolist()
        rows.append(("".join(tokens), selected, None, 0, 0))
    report = selection_entropy(toy(rows)).iloc[0]
    assert report["entropy"] == pytest.approx(report["random_entropy"], abs=0.05)
    assert report["entropy"] == pytest.approx(4.0, abs=0.05)


def test_entropy_can_pool_the_classes_of_a_run():
    frame = toy([("abcd", [0], None, 0, 0), ("abcd", [1], None, 1, 1)])
    per_class = selection_entropy(frame)
    pooled = selection_entropy(frame, per_class=False).iloc[0]
    assert list(per_class["entropy"]) == [0.0, 0.0]
    assert pooled["entropy"] == pytest.approx(1.0)
    assert "label" not in selection_entropy(frame, per_class=False).columns


def test_an_empty_selection_has_zero_entropy():
    report = selection_entropy(toy([("abcd", [], None, 0, 0)])).iloc[0]
    assert report["kept"] == 0
    assert report["entropy"] == 0.0


def test_the_analyzer_reports_entropy(monkeypatch):
    frame = toy([("abcd", [0], None, 0, 0)])
    monkeypatch.setattr(codes.PredictionAnalyzer, "analyze", lambda self: frame.copy())
    assert SelectionCodeAnalyzer("anywhere").entropy().iloc[0]["entropy"] == 0.0
    monkeypatch.setattr(
        codes.PredictionAnalyzer, "analyze", lambda self: pd.DataFrame()
    )
    assert SelectionCodeAnalyzer("anywhere").entropy().empty
