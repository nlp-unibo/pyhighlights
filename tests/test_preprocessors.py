from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.leakage import LeakageDetector
from pyhighlights.components.loaders import HateXplainLoader, HotelLoader
from pyhighlights.components.preprocessors import (
    AnnotationAggregator,
    LeakageRemover,
    Pipeline,
    remove_leakage,
)
from pyhighlights.configurations.keys import (
    HATEXPLAIN_AGGREGATOR,
    HATEXPLAIN_PIPELINE,
    LEAKAGE_REMOVER,
    PIPELINE,
)
from tests.corpora import hatexplain, r2a

HATEXPLAIN_LABELS = ("hatespeech", "normal", "offensive")


def hotel_splits(tmp_path: Path):
    return HotelLoader(url=r2a(tmp_path), directory=tmp_path / "cache").load()


def aggregator(**kwargs) -> AnnotationAggregator:
    return AnnotationAggregator(labels=HATEXPLAIN_LABELS, **kwargs)


def test_leakage_remover_protects_the_annotated_split(tmp_path):
    splits = hotel_splits(tmp_path)
    remover = LeakageRemover()
    repaired = remover.process(splits)

    # Both annotated rows occur in the distributed training split, which also
    # repeats one of its own rows.
    assert remover.removed == {"train": 2, "val": 0, "test": 0}
    assert len(repaired["test"]) == 2
    assert list(repaired["train"]["text"]) == ["a room with no windows"]
    assert list(repaired["train"]["sample_id"]) == [0]
    assert (LeakageDetector().report(repaired)["ratio"] == 0).all()
    assert LeakageDetector().duplicates(repaired) == {"train": 0, "val": 0, "test": 0}
    # The input is left as it was.
    assert len(splits["train"]) == 3


def test_priority_decides_which_split_gives_a_row_up(tmp_path):
    splits = hotel_splits(tmp_path)
    repaired = remove_leakage(splits)

    assert list(repaired) == list(splits)
    assert repaired["test"].equals(splits["test"])

    # Hand training the annotated rows instead and the test split is the one
    # that loses them.
    reversed_priority = LeakageRemover(priority=("train", "val", "test")).process(
        splits
    )
    assert len(reversed_priority["train"]) == 3
    assert len(reversed_priority["test"]) == 0


def test_aggregator_reduces_labels_and_rationales(tmp_path):
    splits = HateXplainLoader(**hatexplain(tmp_path)).load()
    processed = aggregator().process(splits)

    # p2 has no majority label and is dropped; p3 is normal, so nothing is
    # marked; p1 keeps the token both rationale vectors agree on.
    assert [len(frame) for frame in processed.values()] == [1, 1, 1]
    assert list(processed["train"]["text"]) == ["burn them all"]
    assert processed["train"]["highlights"].iloc[0] == [1, 0, 0]
    assert processed["val"]["highlights"].iloc[0] == [0, 0]
    assert processed["val"]["label"].iloc[0] == 1
    assert "annotator_labels" not in processed["train"].columns

    union = aggregator(rationale="union").process(splits)
    assert union["train"]["highlights"].iloc[0] == [1, 1, 0]
    intersection = aggregator(rationale="intersection").process(splits)
    assert intersection["train"]["highlights"].iloc[0] == [1, 0, 0]

    kept = aggregator(ties="keep").process(splits)
    assert len(kept["train"]) == 2


def test_aggregator_leaves_an_already_reduced_corpus_alone(tmp_path):
    splits = hotel_splits(tmp_path)
    processed = aggregator().process(splits)

    assert processed["train"].equals(splits["train"])


def test_aggregator_rejects_settings_and_labels_it_does_not_know():
    with pytest.raises(ValueError, match="rationale must be"):
        aggregator(rationale="whatever")
    with pytest.raises(ValueError, match="ties must be"):
        aggregator(ties="whatever")
    with pytest.raises(ValueError, match="unexpected label"):
        aggregator().label(["mystery", "mystery", "normal"])


def test_pipeline_runs_its_steps_in_order(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    splits = HateXplainLoader(**hatexplain(tmp_path)).load()

    pipeline = Pipeline(steps=[HATEXPLAIN_AGGREGATOR, LEAKAGE_REMOVER])
    processed = pipeline.process(splits)

    # Aggregation first: p1 and p4 reduce to the same text, and the leakage
    # step then keeps it in the annotated split only.
    assert len(processed["test"]) == 1
    assert len(processed["train"]) == 0
    assert (LeakageDetector().report(processed)["ratio"] == 0).all()

    assert Pipeline().process(splits) == splits


def test_registered_preprocessors_build(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    assert isinstance(Registry.from_key(LEAKAGE_REMOVER), LeakageRemover)
    assert isinstance(Registry.from_key(HATEXPLAIN_AGGREGATOR), AnnotationAggregator)

    pipeline = Registry.from_key(PIPELINE)
    assert [type(step) for step in pipeline.preprocessors] == [LeakageRemover]

    hatexplain_pipeline = Registry.from_key(HATEXPLAIN_PIPELINE)
    assert [type(step) for step in hatexplain_pipeline.preprocessors] == [
        AnnotationAggregator,
        LeakageRemover,
    ]
