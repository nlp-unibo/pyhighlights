"""MGR: several generators train, one of them is reported.

Training scores every head, because the generators are there to disagree.
Inference and every reported metric read one of them, and which one is
``inference_head`` -- the two paths nothing exercised before this file.
"""

from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import MGR
from pyhighlights.configurations.keys import GRU_BACKBONE, GRU_MGR, MLP_SELECTOR


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def heads_of(model: MGR) -> int:
    return len(model.selectors)


def batch_of(width: int = 6, size: int = 4) -> InputData:
    return InputData(
        features=th.randint(1, 8, (size, width)),
        mask=th.ones((size, width)),
        sample_ids=th.arange(size),
        y_true=th.randint(0, 2, (size,)),
        highlight_true=th.full((size, width), -1),
    )


class RecordingMetric(th.nn.Module):
    """Keeps what it was handed, which is the point of the test below."""

    def __init__(self):
        super().__init__()
        self.name = "recorded"
        self.seen: list = []

    def update(self, values):
        self.seen.append(values["highlight_mask"].detach().clone())

    def compute(self):
        return th.tensor(0.0)

    def reset(self):
        self.seen.clear()


def test_training_scores_every_generator_and_inference_scores_one():
    model = Registry.from_key(GRU_MGR)
    assert isinstance(model, MGR)
    batch = batch_of()

    heads = len(model.selectors)
    assert heads >= 2
    assert model.training_forward(batch).class_logits.shape[1] == heads
    assert model.validation_forward(batch).class_logits.shape[1] == 1
    assert model.test_forward(batch).class_logits.shape[1] == 1


def test_inference_reads_the_head_it_was_configured_with():
    model = Registry.from_key(GRU_MGR, inference_head=1)
    model.eval()
    batch = batch_of()

    reported = model.test_forward(batch)
    named = model.forward_one_head(batch, selector_idx=1)
    other = model.forward_one_head(batch, selector_idx=0)

    assert th.equal(reported.highlight_mask, named.highlight_mask)
    assert not th.equal(named.highlight_logits, other.highlight_logits)
    with pytest.raises(ValueError, match="outside the generator range"):
        model.forward_one_head(batch, selector_idx=heads_of(model))


def test_the_metrics_score_the_inference_head_and_not_the_first():
    """A training batch carries every head; the reported metric takes one.

    The slice is what makes a metric comparable with the number inference
    reports, and it is silent when it picks the wrong head: every shape still
    fits.
    """
    model = Registry.from_key(GRU_MGR, inference_head=1)
    metric = RecordingMetric()
    model.val_metrics = th.nn.ModuleList([metric])
    batch = batch_of()
    output = model.training_forward(batch)

    model.update_metrics("val", batch, output)

    assert len(metric.seen) == 1
    assert th.equal(metric.seen[0], output.highlight_mask[:, 1])

    # A batch that already carries one head is scored as it arrives: that is
    # every evaluation batch, since inference reports one head to begin with.
    metric.reset()
    single = model.test_forward(batch)
    model.update_metrics("val", batch, single)
    assert th.equal(metric.seen[0], single.highlight_mask[:, 0])


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"predictor_backbone": None}, "separate predictor backbone"),
        (
            {
                "selector_backbones": GRU_BACKBONE,
                "selectors": MLP_SELECTOR,
            },
            "at least two generators",
        ),
        ({"inference_head": 9}, "outside the generator range"),
        ({"loss_reduction": "median"}, "loss_reduction must be"),
    ],
)
def test_the_arrangement_mgr_needs_is_refused_when_it_is_not_there(overrides, message):
    """Each is a run that would train something other than MGR."""
    with pytest.raises(ValueError, match=message):
        Registry.from_key(GRU_MGR, **overrides)
