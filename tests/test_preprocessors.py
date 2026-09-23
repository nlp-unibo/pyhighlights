from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry
from pydantic import ValidationError

import pyhighlights
from pyhighlights.components.data import COLUMNS
from pyhighlights.components.leakage import LeakageDetector
from pyhighlights.components.loaders import HateXplainLoader, HotelLoader
from pyhighlights.components.preprocessors import (
    AnnotationAggregator,
    ClassWeights,
    LabelMapper,
    LeakageRemover,
    LengthFilter,
    Pipeline,
    class_weights,
    remove_leakage,
)
from pyhighlights.configurations.keys import (
    HATEXPLAIN_AGGREGATOR,
    HATEXPLAIN_PIPELINE,
    LEAKAGE_REMOVER,
    PIPELINE,
)
from pyhighlights.configurations.preprocessors import ClassWeightsConfig
from tests.corpora import UNPINNED, hatexplain, r2a

HATEXPLAIN_LABELS = ("hatespeech", "normal", "offensive")


def hotel_splits(tmp_path: Path):
    return HotelLoader(
        url=r2a(tmp_path), directory=tmp_path / "cache", **UNPINNED
    ).load()


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
    assert LeakageDetector().repeats(repaired) == {"train": 0, "val": 0, "test": 0}
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


def test_aggregator_reduces_labels_and_highlights(tmp_path):
    splits = HateXplainLoader(**hatexplain(tmp_path)).load()
    processed = aggregator().process(splits)

    # p2 has no majority label and is dropped; p3 is normal, so nothing is
    # marked; p1 keeps the token both highlight vectors agree on.
    assert [len(frame) for frame in processed.values()] == [1, 1, 1]
    assert list(processed["train"]["text"]) == ["burn them all"]
    assert processed["train"]["highlights"].iloc[0] == [1, 0, 0]
    assert processed["val"]["highlights"].iloc[0] == [0, 0]
    assert processed["val"]["label"].iloc[0] == 1
    assert "annotator_labels" not in processed["train"].columns

    union = aggregator(highlights="union").process(splits)
    assert union["train"]["highlights"].iloc[0] == [1, 1, 0]
    intersection = aggregator(highlights="intersection").process(splits)
    assert intersection["train"]["highlights"].iloc[0] == [1, 0, 0]

    kept = aggregator(ties="keep").process(splits)
    assert len(kept["train"]) == 2


def test_a_tie_is_two_labels_sharing_the_top_count():
    """Not "the top count is one", which is only the same thing at three.

    HateXplain has three annotators, so the two coincide there and the defect
    stayed invisible. They come apart in both directions:

    - four annotators splitting 2-2 have a top count of *two*, so no tie was
      detected and the label was decided by whichever one `Counter` happened
      to order first -- silently, with ``ties="drop"`` not firing
    - a single annotator has a top count of *one*, so every row of a
      single-annotator corpus was dropped as a tie
    """
    voter = aggregator()

    # A real majority is untouched, whatever the panel size.
    assert voter.label(["normal", "normal", "hatespeech"]) == 1
    assert voter.label(["normal", "normal", "hatespeech", "offensive"]) == 1

    # Three splitting three ways: the case that already worked.
    assert voter.label(["normal", "hatespeech", "offensive"]) is None
    # Four splitting 2-2: the case that did not.
    assert voter.label(["normal", "normal", "hatespeech", "hatespeech"]) is None
    # One annotator has nobody to tie with.
    assert voter.label(["hatespeech"]) == 0


def test_keeping_a_tie_still_resolves_it_by_annotator_order():
    """``ties="keep"`` is the other half of the contract, and it has to hold
    for an even split as much as for an odd one."""
    voter = aggregator(ties="keep")

    assert voter.label(["hatespeech", "hatespeech", "normal", "normal"]) == 0
    assert voter.label(["normal", "normal", "hatespeech", "hatespeech"]) == 1
    assert voter.label(["offensive"]) == 2


def test_aggregator_leaves_an_already_reduced_corpus_alone(tmp_path):
    splits = hotel_splits(tmp_path)
    processed = aggregator().process(splits)

    assert processed["train"].equals(splits["train"])


def test_aggregator_rejects_settings_and_labels_it_does_not_know():
    with pytest.raises(ValueError, match="highlights must be"):
        aggregator(highlights="whatever")
    with pytest.raises(ValueError, match="ties must be"):
        aggregator(ties="whatever")
    with pytest.raises(ValueError, match="unexpected label"):
        aggregator().label(["mystery", "mystery", "normal"])
    # Without `labels` a vote is a class index already, and a name is a
    # configuration that forgot to list them rather than a bad integer.
    with pytest.raises(ValueError, match="name the corpus's labels"):
        AnnotationAggregator().label(["normal"])


def test_aggregator_keeps_a_column_outside_the_standard_ones():
    """`knowledge` is an extra column, and reducing annotators is no reason to
    lose it: the split would then weight no links and say nothing about it."""
    frame = pd.DataFrame(
        {
            "sample_id": [0],
            "text": ["a b"],
            "tokens": [["a", "b"]],
            "label": [None],
            "highlights": [None],
            "knowledge": [[1]],
            "annotator_labels": [["normal", "normal"]],
            "annotator_highlights": [[[1, 0], [1, 1]]],
        }
    )

    processed = aggregator().process({"train": frame})["train"]

    assert list(processed.columns) == [*COLUMNS, "knowledge"]
    assert processed["knowledge"].tolist() == [[1]]
    assert processed["label"].tolist() == [1]


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


def test_length_filter_drops_rows_rather_than_truncating_them():
    import pandas as pd

    splits = {
        "train": pd.DataFrame(
            {"tokens": [["a"], ["a", "b", "c"], ["a", "b"]], "label": [0, 1, 0]}
        )
    }
    filtered = LengthFilter(max_length=2).process(splits)

    # Truncating would keep the row and lose the tokens the annotation covers.
    assert [len(tokens) for tokens in filtered["train"]["tokens"]] == [1, 2]
    assert LengthFilter(max_length=2).process(splits)["train"].index.tolist() == [0, 1]

    dropper = LengthFilter(max_length=2)
    dropper.process(splits)
    assert dropper.removed == {"train": 1}

    with pytest.raises(ValueError, match="max_length must be positive"):
        LengthFilter(max_length=0)


def test_label_mapper_rewrites_resolved_labels_and_per_annotator_votes():
    import pandas as pd

    splits = {
        "train": pd.DataFrame(
            {
                "label": ["hatespeech", "offensive", "normal"],
                "annotator_labels": [
                    ["hatespeech", "offensive", "normal"],
                    ["normal", "normal", "offensive"],
                    ["normal", "normal", "normal"],
                ],
            }
        )
    }
    mapping = {"offensive": "normal"}

    resolved = LabelMapper(mapping).process(splits)
    assert resolved["train"]["label"].tolist() == ["hatespeech", "normal", "normal"]

    # Collapsing before the vote is counted is the point: this post has no
    # majority over three classes and a clear one over two.
    votes = LabelMapper(mapping, column="annotator_labels").process(splits)
    assert votes["train"]["annotator_labels"][0] == [
        "hatespeech",
        "normal",
        "normal",
    ]
    aggregated = AnnotationAggregator(
        labels=["hatespeech", "normal"], annotator_highlights="missing"
    )
    assert aggregated.label(votes["train"]["annotator_labels"][0]) == 1

    # The input is left alone, and a class the mapping never names survives.
    assert splits["train"]["label"].tolist()[1] == "offensive"

    with pytest.raises(KeyError, match="no column missing"):
        LabelMapper(mapping, column="missing").process(splits)
    with pytest.raises(ValueError, match="at least one entry"):
        LabelMapper({})


def test_class_weights_invert_frequency():
    # Three of one class and one of the other: the rare one costs three times
    # what the common one does, and the two average to 1.
    assert class_weights([0, 0, 0, 1]) == pytest.approx([4 / 6, 4 / 2])
    assert class_weights([0, 1]) == [1.0, 1.0]


def test_class_weights_count_the_classes_the_model_has():
    assert len(class_weights([0, 1, 1], classes=2)) == 2
    # Absent from the split, so there is no frequency to invert: a weight of
    # zero or of infinity would both train something the corpus never showed.
    with pytest.raises(ValueError, match="no examples"):
        class_weights([0, 1, 1], classes=3)


def test_class_weights_refuse_what_is_not_a_class_index():
    with pytest.raises(ValueError, match="at least one label"):
        class_weights([])
    with pytest.raises(ValueError, match="non-negative"):
        class_weights([0, -1])


def test_class_weights_refuse_a_count_the_split_contradicts():
    """A class count smaller than the labels is not a count, and said so late.

    `classes=1` over a binary split returned `[2.0]` -- one weight, the other
    class dropped without a word -- and a non-positive count returned `[]`.
    Either one reaches training as a weight vector shorter than the model's
    output layer, and fails there about a shape.
    """
    with pytest.raises(ValueError, match="label 1 is not a class of 1"):
        class_weights([0, 1], classes=1)
    for impossible in (0, -1):
        with pytest.raises(ValueError, match="at least one class"):
            class_weights([0], classes=impossible)

    # And the configuration refuses it before a split is ever read.
    with pytest.raises(ValidationError):
        ClassWeightsConfig(classes=0)


def test_the_class_weights_step_reads_a_split_and_changes_nothing():
    splits = {
        "train": pd.DataFrame({"label": [0, 0, 0, 1], "text": list("abcd")}),
        "test": pd.DataFrame({"label": [0, 1], "text": list("ef")}),
    }
    step = ClassWeights()
    processed = step.process(splits)

    assert step.weights == pytest.approx([4 / 6, 4 / 2])
    assert step.counts == {0: 3, 1: 1}
    for name, frame in splits.items():
        pd.testing.assert_frame_equal(processed[name], frame)


def test_the_class_weights_step_weighs_the_split_it_was_given():
    splits = {"val": pd.DataFrame({"label": [0, 1, 1, 1]})}
    step = ClassWeights(split="val")
    assert step.process(splits)["val"] is splits["val"]
    assert step.weights == pytest.approx([4 / 2, 4 / 6])

    with pytest.raises(KeyError, match="'train'"):
        ClassWeights().process(splits)
