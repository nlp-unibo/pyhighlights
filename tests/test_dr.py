"""DR: the predictor's rate is the selection rate, rewritten every step.

The paper restrains the predictor's Lipschitz constant by giving it the
selector's rate scaled by what fraction of the input the selection kept. The
scaling is dynamic, so what has to hold is that it follows the mask, that it is
floored, that it is written from the rate the group was built at rather than
from what the last step left behind, that a step is the unit rather than a
batch, and that an evaluation pass moves nothing.
"""

from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import SPPOutput
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import GRU_DR, TOY

BASE_LR = 1e-3


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def batch_of(width: int = 4) -> InputData:
    return InputData(
        features=th.zeros((1, width), dtype=th.long),
        mask=th.ones((1, width)),
        sample_ids=th.zeros(1, dtype=th.long),
        y_true=th.zeros(1, dtype=th.long),
        highlight_true=th.full((1, width), -1),
    )


def output_keeping(kept: int, width: int = 4) -> SPPOutput:
    mask = th.zeros((1, 1, width))
    mask[:, :, :kept] = 1.0
    return SPPOutput(
        class_logits=th.zeros((1, 1, 2)),
        highlight_logits=th.zeros((1, 1, width, 2)),
        highlight_mask=mask,
    )


def step_after(model, optimizer, *kept: int) -> None:
    """One optimizer step, preceded by a training batch for each ``kept``."""
    for one in kept:
        model.record(
            split="train",
            batch=batch_of(),
            output_data=output_keeping(one),
            total_loss=th.zeros(()),
            losses={},
        )
    model.on_before_optimizer_step(optimizer)


def rates(model, optimizer):
    predictor = [
        group["lr"]
        for index, group in enumerate(optimizer.param_groups)
        if index in model.predictor_rates
    ]
    selector = [
        group["lr"]
        for index, group in enumerate(optimizer.param_groups)
        if index not in model.predictor_rates
    ]
    return predictor, selector


def test_the_predictor_trains_at_the_rate_of_what_it_reads():
    model = Registry.from_key(GRU_DR)
    optimizer = model.configure_optimizers()

    step_after(model, optimizer, 1)
    predictor, selector = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.25]
    assert selector == [BASE_LR]

    # Half the input the next step: read from the base rate, not from the
    # quarter the last one left, which would compound to a sixteenth.
    step_after(model, optimizer, 2)
    predictor, _ = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.5]


def test_a_selection_that_keeps_almost_nothing_hits_the_floor():
    model = Registry.from_key(GRU_DR, scale_floor=0.5)
    optimizer = model.configure_optimizers()

    step_after(model, optimizer, 1)
    predictor, _ = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.5]


def test_accumulated_batches_are_one_rate_for_one_step():
    """A step is the unit, not a batch: four batches, one rate, their mean."""
    model = Registry.from_key(GRU_DR)
    optimizer = model.configure_optimizers()

    step_after(model, optimizer, 1, 1, 3, 3)
    predictor, _ = rates(model, optimizer)
    assert predictor == [pytest.approx(BASE_LR * 0.5)]
    assert model.batch_rates == []


def test_both_halves_of_a_split_predictor_group_are_scaled():
    """``encoder_lr`` splits the predictor in two; the scale applies to both."""
    model = Registry.from_key(GRU_DR, encoder_lr=2e-5)
    optimizer = model.configure_optimizers()
    assert len(model.predictor_rates) == 2

    step_after(model, optimizer, 1)
    predictor, selector = rates(model, optimizer)
    assert sorted(predictor) == [2e-5 * 0.25, BASE_LR * 0.25]
    assert sorted(selector) == [2e-5, BASE_LR]


def test_a_restored_optimizer_is_scaled_from_the_built_rate():
    """`load_state_dict` replaces `param_groups`, and restores a scaled rate.

    So a resumed run must address its groups by index and keep scaling from
    the rate the registration built, not from what the last step wrote.
    """
    model = Registry.from_key(GRU_DR)
    optimizer = model.configure_optimizers()
    step_after(model, optimizer, 1)
    state = optimizer.state_dict()

    resumed = Registry.from_key(GRU_DR)
    restored = resumed.configure_optimizers()
    restored.load_state_dict(state)
    assert rates(resumed, restored)[0] == [BASE_LR * 0.25]

    step_after(resumed, restored, 2)
    predictor, selector = rates(resumed, restored)
    assert predictor == [BASE_LR * 0.5]
    assert selector == [BASE_LR]


def test_an_untrained_model_says_so_rather_than_training_at_one_rate():
    model = Registry.from_key(GRU_DR)
    with pytest.raises(RuntimeError):
        model.on_before_optimizer_step(th.optim.Adam(model.parameters()))


def test_evaluation_does_not_move_a_rate():
    model = Registry.from_key(GRU_DR)
    optimizer = model.configure_optimizers()
    before = [group["lr"] for group in optimizer.param_groups]

    for split in ("val", "test"):
        model.record(
            split=split,
            batch=batch_of(),
            output_data=output_keeping(1),
            total_loss=th.zeros(()),
            losses={},
        )

    assert model.batch_rates == []
    assert [group["lr"] for group in optimizer.param_groups] == before


def test_the_rate_is_written_inside_a_real_training_loop(tmp_path):
    """The hook has to fire while Lightning owns the step, not only when called.

    Under accumulation too, where Lightning runs several batches per step --
    the case a hand-driven call cannot show.
    """
    task = SPPTask(
        loader=TOY,
        model=GRU_DR,
        save_path=str(tmp_path),
        batch_size=8,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    model = task.build_model()
    loaders = task.loaders(task.splits())

    trainer = L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=4,
        accumulate_grad_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    trainer.fit(model, loaders["train"])

    optimizer = trainer.optimizers[0]
    predictor, selector = rates(model, optimizer)
    assert selector == [BASE_LR]
    assert predictor[0] < BASE_LR
    assert predictor[0] >= BASE_LR * model.scale_floor
    # Consumed by the steps, not piling up across the epoch.
    assert model.batch_rates == []
