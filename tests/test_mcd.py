"""MCD: the highlight has to predict what the whole input predicts.

The predictor is trained on both passes and the generator is trained to make
them agree, which is what the two phases are for. These tests are about the
phases and about the pass a validation split is scored on, since that pass
runs no phase at all.
"""

from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import MCD, PhasedSPP
from pyhighlights.configurations.keys import GRU_MCD


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


def test_evaluation_scores_every_group_at_once():
    """A validation batch belongs to no phase, so it is scored by all of them.

    `compute_loss` is the only path an evaluation split takes, and until this
    test nothing ran it: every MCD test trained with `limit_val_batches=0`.
    """
    model = Registry.from_key(GRU_MCD)
    assert isinstance(model, MCD) and isinstance(model, PhasedSPP)
    batch = batch_of()

    total, losses = model.compute_loss(batch, model(batch))

    assert set(losses) == {
        "sparsity",
        "contiguity",
        "classification",
        "full_classification",
        "discrepancy",
    }
    assert th.isfinite(total)


def test_a_validation_epoch_reports_the_loss_it_scored():
    model = Registry.from_key(GRU_MCD)
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    loader = DataLoader([batch_of()], batch_size=None)

    trainer.fit(model, train_dataloaders=loader, val_dataloaders=loader)

    assert th.isfinite(trainer.callback_metrics["val_loss"])
    assert "val_discrepancy" in trainer.callback_metrics


def test_more_than_one_head_is_refused():
    model = Registry.from_key(GRU_MCD)
    batch = batch_of()
    output = model(batch)
    doubled = type(output)(
        class_logits=output.class_logits.repeat(1, 2, 1),
        highlight_logits=output.highlight_logits.repeat(1, 2, 1, 1),
        highlight_mask=output.highlight_mask.repeat(1, 2, 1),
    )

    with pytest.raises(ValueError, match="exactly one head"):
        model.compute_loss(batch, doubled)


def test_a_phased_model_takes_exactly_one_generator():
    """The phases alternate between one generator and one predictor."""
    from pyhighlights.configurations.keys import GRU_BACKBONE, MLP_SELECTOR

    with pytest.raises(ValueError, match="MCD requires exactly one selector"):
        Registry.from_key(
            GRU_MCD,
            selector_backbones=[GRU_BACKBONE, GRU_BACKBONE],
            selectors=[MLP_SELECTOR, MLP_SELECTOR],
        )


def test_the_highlight_pass_is_what_mcd_trains_on():
    """Unlike MRD's, which is scored outside the graph."""
    model = Registry.from_key(GRU_MCD)
    output, values = model.phase_forward(batch_of(), detach_selection=False)

    assert output.class_logits.requires_grad
    assert "full_class_logits" in values


def test_the_predictor_phase_also_steps_the_generator():
    """The shared terms bind to the selection the generator produced.

    The predictor reads a detached copy, so classification stays on the
    predictor's side, but sparsity and contiguity reach the generator here as
    well as in its own phase -- and `training_step` steps the generator's
    optimizer in both. Asserted rather than assumed: it is the reason the
    generator takes two steps per batch on those terms, and a change to it is
    a change to what the model optimizes.
    """
    model = Registry.from_key(GRU_MCD)
    batch = batch_of()

    total, losses, _ = model.predictor_phase_loss(batch)
    total.backward()
    selector_gradient = model.selectors[0].selector[-1].weight.grad
    predictor_gradient = model.predictor.predictor[-1].weight.grad

    assert set(losses) == {
        "sparsity",
        "contiguity",
        "classification",
        "full_classification",
    }
    assert predictor_gradient is not None and predictor_gradient.abs().sum() > 0
    assert selector_gradient is not None and selector_gradient.abs().sum() > 0
