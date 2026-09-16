from pathlib import Path
from typing import List

import pytest
import torch as th
import torchmetrics
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, Registry

import pyhighlights
from pyhighlights.configurations.keys import (
    ACCURACY_METRIC,
    CLASS_F1_METRIC,
    F1_METRIC,
    HIGHLIGHT_F1_METRIC,
    HIGHLIGHT_IOU_METRIC,
    MULTICLASS_ACCURACY_METRIC,
    MULTICLASS_F1_METRIC,
    HIGHLIGHT_PRECISION_METRIC,
    HIGHLIGHT_RECALL_METRIC,
    SELECTION_RATE_METRIC,
    SELECTION_SIZE_METRIC,
    SELECTION_SPANS_METRIC,
)
from pyhighlights.utility.metrics import (
    BinaryHighlightF1Score,
    BinaryHighlightIoU,
    BinaryHighlightPrecision,
    BinaryHighlightRecall,
    BoundMetric,
    ClassF1Score,
    SelectionRate,
    SelectionSize,
    SelectionSpans,
    build_metrics,
)


class MulticlassMetricConfig(Configuration):
    num_classes: int = Param(3)


class BoundMetricConfig(Configuration):
    name: str = Param("accuracy")
    metric: RegistrationKey = Param(
        RegistrationKey(name="class_binding", namespace="tests")
    )
    inputs: List[str] = Param(["class_logits", "y_true"])


def test_build_bound_metrics_from_registered_metrics():
    Registry.initialize()
    highlight_f1 = Registry.register_configuration(
        config=Configuration.default(),
        name="highlight_f1",
        namespace="tests",
        component="pyhighlights.utility.metrics.BinaryHighlightF1Score",
    )
    class_binding = Registry.register_configuration(
        config=MulticlassMetricConfig.default(),
        name="class_binding",
        namespace="tests",
        component="torchmetrics.classification.MulticlassAccuracy",
    )
    binding = Registry.register_configuration(
        config=BoundMetricConfig.default(),
        name="binding",
        namespace="tests",
        component="pyhighlights.utility.metrics.BoundMetric",
    )
    Registry.dag_resolution()

    accuracy = BoundMetric(name="accuracy", metric=class_binding)
    highlight = BoundMetric(
        name="highlight_f1",
        metric=highlight_f1,
        inputs=["highlight_mask", "highlight_true"],
    )
    assert isinstance(accuracy.metric, torchmetrics.Metric)
    assert accuracy.inputs == ["class_logits", "y_true"]

    values = {
        "class_logits": th.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]),
        "y_true": th.tensor([0, 1]),
        "highlight_mask": th.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "highlight_true": th.tensor([[1, 0], [1, -1]]),
    }
    accuracy.update(values)
    highlight.update(values)

    assert accuracy.compute() == 1.0
    assert highlight.compute() == 1.0

    highlight.reset()
    with pytest.raises(KeyError, match="highlight_true"):
        highlight.update({"highlight_mask": values["highlight_mask"]})

    built = build_metrics([binding])
    assert isinstance(built, th.nn.ModuleList)
    assert [metric.name for metric in built] == ["accuracy"]
    assert len(build_metrics(None)) == 0


def test_highlight_metrics_ignore_unlabelled_positions():
    preds = th.tensor([[1.0, 1.0, 0.0]])
    target = th.tensor([[1, 0, -1]])

    f1 = BinaryHighlightF1Score()
    f1.update(preds, target)
    iou = BinaryHighlightIoU()
    iou.update(preds, target)

    assert f1.compute() == pytest.approx(2 / 3)
    assert iou.compute() == pytest.approx(0.5)


def test_a_highlight_score_is_nan_when_nothing_was_asked_of_it():
    """Two ways to divide by zero, and both mean the same thing.

    An update whose every position is a true negative leaves all three
    counters at zero, exactly as never updating does. It is not a model that
    scored badly, so it is not 0.0, and it is not a model that scored
    perfectly for selecting nothing, so it is not 1.0.
    """
    # Annotated, seen, and every position a true negative: the corpus says
    # nothing here is a highlight and the model marked nothing.
    preds = th.tensor([[0.0, 0.0, 0.0]])
    target = th.tensor([[0, 0, -1]])

    f1 = BinaryHighlightF1Score()
    f1.update(preds, target)
    iou = BinaryHighlightIoU()
    iou.update(preds, target)

    assert (f1.tp, f1.fp, f1.fn) == (0, 0, 0)
    assert th.isnan(f1.compute())
    assert th.isnan(iou.compute())

    # One marked position is enough to define both again.
    f1.update(th.tensor([[1.0, 0.0, 0.0]]), th.tensor([[1, 0, -1]]))
    iou.update(th.tensor([[1.0, 0.0, 0.0]]), th.tensor([[1, 0, -1]]))
    assert f1.compute() == pytest.approx(1.0)
    assert iou.compute() == pytest.approx(1.0)


def test_selection_metrics_average_over_samples():
    # `target` is the padding mask: 1 is a real token, 0 is padding.
    preds = th.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    target = th.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])

    rate = SelectionRate()
    rate.update(preds, target)
    size = SelectionSize()
    size.update(preds, target)

    # First sample keeps both of its two tokens, the second one of three.
    assert rate.compute() == pytest.approx((1.0 + 1 / 3) / 2)
    assert size.compute() == pytest.approx(1.5)


def test_a_selection_metric_reduces_the_batch_at_once(monkeypatch):
    """The batch is one pair of masked sums, not a Python loop over rows.

    Two updates cost 2.36 ms against a 54 ms training step before this, all of
    it interpreter overhead. What has to survive vectorising is the treatment
    of a row with no token at all: it has no rate, so it is left out of the
    denominator rather than counted as a zero.
    """
    preds = th.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    # The third row is pure padding: nothing there could have been kept.
    target = th.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0]])

    rate = SelectionRate()
    rate.update(preds, target)
    size = SelectionSize()
    size.update(preds, target)

    assert rate.samples == 2
    assert rate.compute() == pytest.approx((1.0 + 1 / 3) / 2)
    assert size.compute() == pytest.approx(1.5)

    # A padded position is dropped whatever it holds. Multiplying by the mask
    # instead would carry `0 * nan` into the sum and poison the whole batch,
    # where the loop never looked at that position at all.
    poisoned = SelectionRate()
    poisoned.update(
        th.tensor([[1.0, 1.0, float("nan")], [1.0, 0.0, 0.0]]),
        th.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]]),
    )
    assert poisoned.compute() == pytest.approx((1.0 + 1 / 3) / 2)

    # And `reduce` sees the whole batch once, not one row at a time.
    calls = []
    original = SelectionRate.reduce
    monkeypatch.setattr(
        SelectionRate,
        "reduce",
        lambda self, selected, length: (
            calls.append(len(selected)) or original(self, selected, length)
        ),
    )
    SelectionRate().update(preds, target)
    assert calls == [2]


def test_a_selection_rate_ignores_the_padding_it_could_not_have_kept():
    """The denominator is the document, not the widest row in the batch.

    This is the defect that made every reported ``selection_rate`` too low.
    The metric excluded positions equal to an ``ignore_index`` of -1, but the
    registered binding hands it ``mask``, which is 0 for padding and never -1,
    so padding stayed in the denominator. A selector keeping a fifth of a
    short clause reported a fifteenth of a padded batch.
    """
    # One four-token document in a batch padded to twelve, one token kept.
    preds = th.tensor([[1.0] + [0.0] * 11])
    target = th.tensor([[1.0] * 4 + [0.0] * 8])

    rate = SelectionRate()
    rate.update(preds, target)
    size = SelectionSize()
    size.update(preds, target)

    assert rate.compute() == pytest.approx(0.25)
    assert size.compute() == pytest.approx(1.0)

    # Widening the batch must not move the rate: same document, more padding.
    wider = SelectionRate()
    wider.update(
        th.tensor([[1.0] + [0.0] * 39]),
        th.tensor([[1.0] * 4 + [0.0] * 36]),
    )
    assert wider.compute() == pytest.approx(rate.compute())


def test_registered_metrics_score_the_fields_they_name():
    """The registered set is what a task hands a model, so it must bind."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        # Two classes, one logit each: the classification metrics are
        # registered as multiclass even for a binary corpus.
        "class_logits": th.tensor([[2.0, 0.0], [0.0, 2.0]]),
        "y_true": th.tensor([0, 1]),
        "highlight_mask": th.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "highlight_true": th.tensor([[1, 0], [1, -1]]),
        "mask": th.tensor([[1.0, 1.0], [1.0, 1.0]]),
    }
    scores = {}
    for key in (
        ACCURACY_METRIC,
        F1_METRIC,
        HIGHLIGHT_F1_METRIC,
        HIGHLIGHT_IOU_METRIC,
        SELECTION_RATE_METRIC,
        SELECTION_SIZE_METRIC,
    ):
        metric = Registry.from_key(key, expected_type=BoundMetric)
        metric.update(values)
        scores[metric.name] = metric.compute().item()

    assert scores["accuracy"] == pytest.approx(1.0)
    assert scores["f1"] == pytest.approx(1.0)
    # Two annotated positions are marked and both are selected; the third is
    # unannotated and scores nothing.
    assert scores["highlight_f1"] == pytest.approx(1.0)
    assert scores["highlight_iou"] == pytest.approx(1.0)
    assert scores["selection_rate"] == pytest.approx(0.75)
    assert scores["selection_size"] == pytest.approx(1.5)


def test_multiclass_metrics_are_registered_for_hatexplain():
    """HateXplain has three classes; the binary registrations cannot score it."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        "class_logits": th.tensor([[2.0, 0.0, 0.0], [0.0, 0.0, 2.0]]),
        "y_true": th.tensor([0, 1]),
    }
    accuracy = Registry.from_key(MULTICLASS_ACCURACY_METRIC, expected_type=BoundMetric)
    accuracy.update(values)

    assert accuracy.compute().item() == pytest.approx(0.5)

    f1 = Registry.from_key(MULTICLASS_F1_METRIC, expected_type=BoundMetric)
    f1.update(values)
    assert 0.0 <= f1.compute().item() <= 1.0


def test_each_build_gets_its_own_metric_state():
    """Two models must not share a metric: seeds would accumulate each other."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    first = Registry.from_key(ACCURACY_METRIC, expected_type=BoundMetric)
    second = Registry.from_key(ACCURACY_METRIC, expected_type=BoundMetric)
    first.update({"class_logits": th.tensor([[2.0, 0.0]]), "y_true": th.tensor([0])})

    assert first.metric is not second.metric
    assert second.metric.compute().item() == pytest.approx(0.0)


def test_class_f1_reports_the_rare_class_where_macro_reports_the_other_one():
    """Nineteen negatives and one positive, all called negative.

    Macro F1 answers 0.49 -- half of a perfect score on the class that is 95%
    of the rows -- and the model found nothing. The class metric answers 0.0,
    which is the number a skewed corpus is read with.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    values = {
        "class_logits": th.tensor([[2.0, 0.0]] * 20),
        "y_true": th.tensor([0] * 19 + [1]),
    }

    macro = Registry.from_key(F1_METRIC, expected_type=BoundMetric)
    macro.update(values)
    assert macro.compute().item() == pytest.approx(0.4872, abs=1e-4)

    unfair = Registry.from_key(CLASS_F1_METRIC, expected_type=BoundMetric)
    unfair.update(values)
    # Reported under the same name: which F1 it is, the manifest's key says.
    assert unfair.name == "f1"
    assert unfair.compute().item() == pytest.approx(0.0)

    found = Registry.from_key(CLASS_F1_METRIC, expected_type=BoundMetric)
    found.update(
        {
            "class_logits": th.tensor([[2.0, 0.0]] * 19 + [[0.0, 2.0]]),
            "y_true": th.tensor([0] * 19 + [1]),
        }
    )
    assert found.compute().item() == pytest.approx(1.0)


def test_class_f1_refuses_a_class_it_cannot_score():
    with pytest.raises(ValueError, match="pos_label"):
        ClassF1Score(pos_label=2, num_classes=2)


def test_precision_and_recall_split_what_the_f1_averages():
    """A selector that keeps too little is precise and does not recall.

    The point of reporting the pair: F1 alone reads as one slightly worse
    model, where the two say the selection changed shape.
    """
    # Four annotated positions, of which the model marks one, and nothing else.
    preds = th.tensor([[1.0, 0.0, 0.0, 0.0]])
    target = th.tensor([[1, 1, 1, 1]])

    precision = BinaryHighlightPrecision()
    precision.update(preds, target)
    recall = BinaryHighlightRecall()
    recall.update(preds, target)

    assert precision.compute() == pytest.approx(1.0)
    assert recall.compute() == pytest.approx(0.25)

    # And the other way: everything marked, only one of them annotated.
    precision = BinaryHighlightPrecision()
    recall = BinaryHighlightRecall()
    preds = th.tensor([[1.0, 1.0, 1.0, 1.0]])
    target = th.tensor([[1, 0, 0, 0]])
    precision.update(preds, target)
    recall.update(preds, target)

    assert precision.compute() == pytest.approx(0.25)
    assert recall.compute() == pytest.approx(1.0)


def test_precision_and_recall_are_nan_on_their_own_empty_denominators():
    """The two divide by zero on different splits, and neither scores it 0.

    Precision has no denominator when the model marked nothing; recall has none
    when the corpus annotates nothing. A split can define one and not the other,
    which is why they do not share :class:`BinaryHighlightF1Score`'s condition.
    """
    # Annotated, and the model marked none of it: recall is 0, precision is nan.
    precision = BinaryHighlightPrecision()
    recall = BinaryHighlightRecall()
    preds = th.tensor([[0.0, 0.0]])
    target = th.tensor([[1, 1]])
    precision.update(preds, target)
    recall.update(preds, target)

    assert th.isnan(precision.compute())
    assert recall.compute() == pytest.approx(0.0)

    # Unannotated, and the model marked something: precision is 0, recall nan.
    precision = BinaryHighlightPrecision()
    recall = BinaryHighlightRecall()
    preds = th.tensor([[1.0, 1.0]])
    target = th.tensor([[0, 0]])
    precision.update(preds, target)
    recall.update(preds, target)

    assert precision.compute() == pytest.approx(0.0)
    assert th.isnan(recall.compute())


def test_selection_spans_counts_runs_not_tokens():
    """Six scattered words and six contiguous ones are the same size.

    Which is the whole reason this is reported: `selection_size` cannot tell a
    phrase from a model keying on punctuation across the clause.
    """
    # Same four tokens kept in each row, in one run and then in three.
    preds = th.tensor([[1.0, 1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 1.0, 0.0, 1.0]])
    target = th.tensor([[1.0, 1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0, 1.0]])

    spans = SelectionSpans()
    spans.update(preds, target)
    size = SelectionSize()
    size.update(preds, target)

    assert spans.compute() == pytest.approx((1 + 3) / 2)
    # The sizes differ by one, and say nothing about the shape.
    assert size.compute() == pytest.approx((4 + 3) / 2)


def test_selection_spans_handles_the_edges_and_the_padding():
    # A run touching column zero is one span, not two: the predecessor of the
    # first column is supplied as "not kept".
    spans = SelectionSpans()
    spans.update(th.tensor([[1.0, 1.0, 0.0]]), th.tensor([[1.0, 1.0, 1.0]]))
    assert spans.compute() == pytest.approx(1.0)

    # Selecting nothing is zero spans; selecting everything is one.
    spans = SelectionSpans()
    spans.update(th.tensor([[0.0, 0.0, 0.0]]), th.tensor([[1.0, 1.0, 1.0]]))
    assert spans.compute() == pytest.approx(0.0)

    spans = SelectionSpans()
    spans.update(th.tensor([[1.0, 1.0, 1.0]]), th.tensor([[1.0, 1.0, 1.0]]))
    assert spans.compute() == pytest.approx(1.0)

    # Padding cannot close a run it is not part of, and a row of pure padding
    # is left out of the denominator like every other selection metric.
    spans = SelectionSpans()
    spans.update(
        th.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 1.0]]),
        th.tensor([[1.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
    )
    assert spans.samples == 1
    assert spans.compute() == pytest.approx(1.0)


def test_the_new_highlight_metrics_are_registered():
    """Each new key builds and scores, which is what a task asking for it needs."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    preds = th.tensor([[1.0, 1.0, 0.0, 0.0]])
    annotation = th.tensor([[1, 0, 1, -1]])
    mask = th.tensor([[1.0, 1.0, 1.0, 0.0]])

    values = {"highlight_mask": preds, "highlight_true": annotation, "mask": mask}
    for key, expected in [
        # Marked positions 0 and 1; position 0 is annotated, 1 is not, 2 is an
        # annotation the model missed, 3 carries none. So tp=1, fp=1, fn=1.
        (HIGHLIGHT_PRECISION_METRIC, 0.5),
        (HIGHLIGHT_RECALL_METRIC, 0.5),
        # Over `mask`, not the annotation: one run of two tokens.
        (SELECTION_SPANS_METRIC, 1.0),
    ]:
        bound = Registry.from_key(key, expected_type=BoundMetric)
        bound.update(values)
        assert bound.compute().item() == pytest.approx(expected), key
