import zipfile
from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.datasets import (
    R2ALoader,
    cache_directory,
    extract,
    remove_leakage,
    to_examples,
)
from pyhighlights.configurations.datasets import R2A

LABELLED = "task\tlabel\ttext\n"
ANNOTATED = "task\tlabel\ttext\trationale\tpred_att\n"


def archive(directory: Path) -> str:
    """A miniature R2A archive: the test split leaks fully into train."""
    leaked = "the location is central but noisy"
    files = {
        "data/oracle/hotel_Location.train": LABELLED
        + f"hotel_Location\t1\t{leaked}\n"
        + "hotel_Location\t0\ta room with no windows\n"
        + "hotel_Location\t1\tclose to the station\n",
        "data/oracle/hotel_Location.dev": LABELLED
        + "hotel_Location\t0\tfar from everything\n",
        # Whitespace differs from the training row on purpose.
        "data/target/hotel_Location.train": ANNOTATED
        + "hotel_Location\t1\tThe  location is central   but noisy\t0 1 0 0 0 0\t0.1\n"
        + "hotel_Location\t1\tclose to the station\t0 0 0 1\t0.2\n",
    }
    path = directory / "mini.zip"
    with zipfile.ZipFile(path, "w") as target:
        for name, content in files.items():
            target.writestr(name, content)
    return path.as_uri()


def loader(tmp_path: Path, **kwargs) -> R2ALoader:
    kwargs.setdefault("remove_leakage", False)
    return R2ALoader(
        task="hotel_Location",
        url=archive(tmp_path),
        directory=tmp_path / "cache",
        **kwargs,
    )


def test_loader_downloads_once_and_parses_the_standard_columns(tmp_path):
    source = loader(tmp_path)
    splits = source.load()

    assert set(splits) == {"train", "val", "test"}
    assert list(splits["train"].columns) == [
        "sample_id",
        "text",
        "tokens",
        "label",
        "highlights",
    ]
    assert len(splits["train"]) == 3
    assert splits["train"]["highlights"].isna().all()

    test = splits["test"]
    assert test["tokens"].iloc[1] == ["close", "to", "the", "station"]
    assert test["highlights"].iloc[1] == [0, 0, 0, 1]
    assert list(test["sample_id"]) == [0, 1]

    # The archive is fetched once; a second load reuses the extracted copy.
    (source.directory / source.archive_name).unlink()
    assert source.read()["test"].equals(test)


def test_leakage_reports_the_share_of_each_split_already_seen(tmp_path):
    report = loader(tmp_path).leakage().set_index(["left", "right"])

    # Every annotated evaluation row occurs in training, one of them only
    # after whitespace normalization.
    assert report.loc[("train", "test"), "ratio"] == 1.0
    assert report.loc[("train", "test"), "overlap"] == 2
    assert report.loc[("train", "val"), "ratio"] == 0.0
    assert report.loc[("test", "train"), "ratio"] == pytest.approx(2 / 3)

    raw = loader(tmp_path).leakage(normalize_keys=False).set_index(["left", "right"])
    assert raw.loc[("train", "test"), "ratio"] == 0.5


def test_frames_convert_to_highlight_examples(tmp_path):
    datasets = loader(tmp_path).datasets()
    example = datasets["test"][1]

    assert len(datasets["train"]) == 3
    assert example.highlights == (0, 0, 0, 1)
    assert datasets["train"][0].highlights is None


def test_misaligned_rationales_are_rejected(tmp_path):
    path = tmp_path / "broken.train"
    path.write_text(ANNOTATED + "hotel_Location\t1\ttwo tokens\t0 1 1\t0.1\n")

    with pytest.raises(ValueError, match="misaligned"):
        R2ALoader.read_file(path)


def test_continuous_labels_are_rejected(tmp_path):
    path = tmp_path / "source.train"
    path.write_text(LABELLED + "beer0\t0.3\tno taste at all\n")

    with pytest.raises(ValueError, match="continuous labels"):
        R2ALoader.read_file(path)


def test_unsafe_archive_members_are_refused(tmp_path):
    path = tmp_path / "escape.zip"
    with zipfile.ZipFile(path, "w") as target:
        target.writestr("../escaped.txt", "nope")

    with pytest.raises(ValueError, match="unsafe archive path"):
        extract(path, tmp_path / "out")


def test_cache_directory_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PYHIGHLIGHTS_CACHE", str(tmp_path / "elsewhere"))
    assert cache_directory() == tmp_path / "elsewhere"

    monkeypatch.delenv("PYHIGHLIGHTS_CACHE")
    assert cache_directory() == Path.home() / ".cache" / "pyhighlights"


def test_registered_loader_builds_with_task_variants(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    source = Registry.from_key(
        R2A, task="beer0", url=archive(tmp_path), directory=str(tmp_path / "cache")
    )

    assert isinstance(source, R2ALoader)
    assert source.task == "beer0"
    with pytest.raises(ValueError, match="task must be one of"):
        R2ALoader(task="wine")


def test_to_examples_needs_the_standard_columns(tmp_path):
    frame = loader(tmp_path).load()["train"].drop(columns=["label"])

    with pytest.raises(AttributeError):
        to_examples(frame)


def test_splits_are_leak_free_by_default(tmp_path):
    source = loader(tmp_path, remove_leakage=True)
    splits = source.load()

    # Both annotated rows occur in the distributed training split; the
    # training split also repeats one of its own rows.
    assert source.removed == {"train": 2, "val": 0, "test": 0}
    assert len(splits["test"]) == 2
    assert len(splits["train"]) == 1
    assert list(splits["train"]["text"]) == ["a room with no windows"]
    assert list(splits["train"]["sample_id"]) == [0]

    assert (source.check_leakage()["ratio"] == 0).all()
    assert source.duplicates() == {"train": 0, "val": 0, "test": 0}


def test_check_leakage_reports_the_offending_pairs(tmp_path):
    with pytest.raises(ValueError, match="share rows above the 0.0 tolerance"):
        loader(tmp_path).check_leakage()

    # The distributed splits leak wholly, so only a full tolerance passes.
    assert loader(tmp_path).check_leakage(tolerance=1.0) is not None


def test_remove_leakage_protects_the_annotated_split(tmp_path):
    frames = loader(tmp_path).load()
    repaired = remove_leakage(frames)

    assert list(repaired) == list(frames)
    assert repaired["test"].equals(frames["test"])

    # Priority is explicit: hand training the annotated rows instead and the
    # test split is the one that gives them up.
    reversed_priority = remove_leakage(frames, priority=("train", "val", "test"))
    assert len(reversed_priority["train"]) == 3
    assert len(reversed_priority["test"]) == 0
