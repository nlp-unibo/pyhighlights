"""MRD: the predictor reads what the highlight left behind, and the full input.

The criterion is the discrepancy between those two. Removing plain noise or a
spurious feature leaves the remainder looking like the whole input; removing
the causal features does not. So the generator *maximizes* that discrepancy,
and the predictor is never trained on the highlight at all -- two things that
make this model read differently from every other one here, and the two the
tests are about.
"""

from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import MRD
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import (
    GRU_MRD,
    REMAINING_DISCREPANCY_LOSS,
    TOY,
)


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def batch_of(width: int = 6, size: int = 4) -> InputData:
    return InputData(
        features=th.randint(1, 8, (size, width)),
        mask=th.ones((size, width)),
        sample_ids=th.arange(size),
        y_true=th.randint(0, 2, (size,)),
        highlight_true=th.full((size, width), -1),
    )


def test_the_criterion_is_maximized_not_minimized():
    """A negative coefficient, which no other loss in the library has."""
    loss = Registry.from_key(REMAINING_DISCREPANCY_LOSS)

    assert loss.coefficient < 0
    assert loss.inputs == ["complement_class_logits", "full_class_logits"]


def test_the_complement_is_what_the_highlight_left():
    model = Registry.from_key(GRU_MRD)
    batch = batch_of()
    output = model(batch)
    highlight = model.aggregator(output).highlight_mask

    kept = model.predict(batch, highlight)
    left = model.predict_complement(batch, highlight)
    everything = model.predict_full(batch)

    assert kept.shape == left.shape == everything.shape
    # Nothing kept and nothing left is the same pass over an empty input.
    empty = th.zeros_like(highlight)
    assert th.allclose(
        model.predict(batch, empty),
        model.predict_complement(batch, th.ones_like(empty)),
    )


def test_the_namespace_carries_both_extra_passes():
    model = Registry.from_key(GRU_MRD)
    batch = batch_of()
    _, values = model.phase_forward(batch, detach_selection=True)

    assert {"complement_class_logits", "full_class_logits"} <= set(values)
    total, losses = model.compute_loss(batch, model(batch))
    assert set(losses) == {
        "sparsity",
        "contiguity",
        "complement_classification",
        "full_classification",
        "remaining_discrepancy",
    }
    assert th.isfinite(total)


def test_the_highlight_pass_trains_nothing():
    """It is scored and reported; no gradient comes back through it."""
    model = Registry.from_key(GRU_MRD)
    output, _ = model.phase_forward(batch_of(), detach_selection=True)

    assert not output.class_logits.requires_grad
    assert output.highlight_mask.requires_grad


def test_the_predictor_phase_leaves_the_generator_to_its_own_terms():
    """Classification reaches the predictor only: the selection is detached."""
    model = Registry.from_key(GRU_MRD)
    batch = batch_of()

    total, _, _ = model.predictor_phase_loss(batch)
    total.backward()
    selector_gradient = model.selectors[0].selector[-1].weight.grad
    predictor_gradient = model.predictor.predictor[-1].weight.grad

    assert predictor_gradient is not None and predictor_gradient.abs().sum() > 0
    # The sparsity and contiguity terms are the generator's share of this
    # phase, as in the reference implementation, and they do reach it.
    assert selector_gradient is not None


def test_the_generator_phase_freezes_the_predictor_and_restores_it():
    model = Registry.from_key(GRU_MRD)
    batch = batch_of()

    total, _, _ = model.generator_phase_loss(batch)
    total.backward()

    assert all(
        parameter.requires_grad
        for parameter in (
            *model.predictor.parameters(),
            *model.predictor_backbone.parameters(),
        )
    )
    assert model.predictor.predictor[-1].weight.grad is None
    assert model.selectors[0].selector[-1].weight.grad is not None


def test_supervision_has_to_name_a_phase():
    with pytest.raises(ValueError, match="rationale_losses"):
        Registry.from_key(GRU_MRD, supervise_highlights=True)


def test_a_shared_backbone_is_refused():
    with pytest.raises(ValueError, match="separate predictor backbone"):
        Registry.from_key(GRU_MRD, predictor_backbone=None)


def test_both_phases_run_under_lightning(tmp_path):
    task = SPPTask(
        loader=TOY,
        model=GRU_MRD,
        save_path=str(tmp_path),
        batch_size=8,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    model = task.build_model()
    assert isinstance(model, MRD)
    loaders = task.loaders(task.splits())
    before = [parameter.detach().clone() for parameter in model.parameters()]

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loaders["train"])

    moved = [
        not th.equal(was, now)
        for was, now in zip(before, model.parameters())
        if now.requires_grad
    ]
    assert any(moved)
