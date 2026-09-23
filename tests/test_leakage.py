from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.leakage import LeakageDetector, duplicates, leakage
from pyhighlights.components.loaders import HotelLoader
from pyhighlights.components.preprocessors import remove_leakage
from pyhighlights.configurations.keys import LEAKAGE_DETECTOR
from tests.corpora import UNPINNED, r2a


def hotel_splits(tmp_path: Path):
    return HotelLoader(
        url=r2a(tmp_path), directory=tmp_path / "cache", **UNPINNED
    ).load()


def test_report_gives_the_share_of_each_split_already_seen(tmp_path):
    splits = hotel_splits(tmp_path)
    report = LeakageDetector().report(splits).set_index(["left", "right"])

    # Every annotated evaluation row occurs in training, one of them only
    # after whitespace normalization.
    assert report.loc[("train", "test"), "ratio"] == 1.0
    assert report.loc[("train", "test"), "overlap"] == 2
    assert report.loc[("train", "val"), "ratio"] == 0.0
    assert report.loc[("test", "train"), "ratio"] == pytest.approx(2 / 3)

    raw = leakage(splits, normalize_keys=False).set_index(["left", "right"])
    assert raw.loc[("train", "test"), "ratio"] == 0.5
    assert duplicates(splits) == {"train": 0, "val": 0, "test": 0}


def test_check_names_the_offending_pairs(tmp_path):
    splits = hotel_splits(tmp_path)

    with pytest.raises(ValueError, match="splits share rows"):
        LeakageDetector().check(splits)

    # A corpus distributed leaky is read rather than refused, which is what
    # `report` is for: it says how much is shared and raises nothing.
    report = LeakageDetector().report(splits).set_index(["left", "right"])
    assert report.loc[("train", "test"), "ratio"] == 1.0


def test_a_row_without_text_is_shared_with_nothing():
    """Two rows a corpus left blank are two unusable rows, not a duplicate."""
    blank = pd.DataFrame({"text": ["a", "", None, " "]})
    splits = {"train": blank, "test": pd.DataFrame({"text": ["", "a"]})}

    report = leakage(splits).set_index(["left", "right"])
    # Only "a" is shared, and the blank test row still counts towards `size`.
    assert report.loc[("train", "test"), "overlap"] == 1
    assert report.loc[("train", "test"), "size"] == 2
    assert duplicates(splits)["train"] == 0

    # `test` is walked first, so it keeps "a" and loses its blank row, and
    # `train` loses the row it shares plus all three of its blanks.
    repaired = remove_leakage(splits)
    assert repaired["test"]["text"].tolist() == ["a"]
    assert repaired["train"]["text"].tolist() == []


def test_registered_detector_builds():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    detector = Registry.from_key(LEAKAGE_DETECTOR)

    assert isinstance(detector, LeakageDetector)
    assert detector.key == "text"
