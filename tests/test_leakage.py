from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.leakage import LeakageDetector, duplicates, leakage
from pyhighlights.components.loaders import HotelLoader
from pyhighlights.configurations.keys import LEAKAGE_DETECTOR
from tests.corpora import r2a


def hotel_splits(tmp_path: Path):
    return HotelLoader(url=r2a(tmp_path), directory=tmp_path / "cache").load()


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

    with pytest.raises(ValueError, match="share rows above the 0.0 tolerance"):
        LeakageDetector().check(splits)

    # The distributed splits leak wholly, so only a full tolerance passes.
    assert LeakageDetector().check(splits, tolerance=1.0) is not None
    assert LeakageDetector(tolerance=1.0).check(splits) is not None

    with pytest.raises(ValueError, match="tolerance must be between 0 and 1"):
        LeakageDetector(tolerance=2.0)


def test_registered_detector_builds():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    detector = Registry.from_key(LEAKAGE_DETECTOR)

    assert isinstance(detector, LeakageDetector)
    assert detector.tolerance == 0.0
