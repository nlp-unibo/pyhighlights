import math

import pandas as pd
import pytest

from pyhighlights.utility.statistics import COLUMNS, describe


def corpus():
    return {
        "train": pd.DataFrame(
            [
                {"tokens": list("abcd"), "highlights": [1, 0, 0, 0]},
                {"tokens": list("abcdef"), "highlights": [0, 1, 1, 0, 0, 0]},
            ]
        ),
        "val": pd.DataFrame(
            [{"tokens": list("abc"), "highlights": None}],
        ),
        "test": pd.DataFrame(columns=["tokens", "highlights"]),
    }


def test_describe_reports_lengths_and_annotation_density():
    report = describe(corpus()).set_index("split")

    assert list(describe({}).columns) == COLUMNS
    assert report.loc["train", "rows"] == 2
    assert report.loc["train", "annotated"] == 2
    assert report.loc["train", "tokens_min"] == 4.0
    assert report.loc["train", "tokens_mean"] == 5.0
    assert report.loc["train", "tokens_p95"] == pytest.approx(5.9)
    assert report.loc["train", "tokens_max"] == 6.0

    # Three marked tokens over the ten the annotated rows hold, and 1.5 marked
    # tokens per row -- the number the rate alone hides.
    assert report.loc["train", "highlight_rate"] == pytest.approx(0.3)
    assert report.loc["train", "highlights_mean"] == pytest.approx(1.5)


def test_a_split_without_annotation_reports_nan_rather_than_zero():
    report = describe(corpus()).set_index("split")

    assert report.loc["val", "annotated"] == 0
    assert math.isnan(report.loc["val", "highlight_rate"])
    assert math.isnan(report.loc["val", "highlights_mean"])
    assert report.loc["val", "tokens_mean"] == 3.0

    assert report.loc["test", "rows"] == 0
    assert math.isnan(report.loc["test", "tokens_mean"])
