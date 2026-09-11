"""DR: the predictor's rate is the selection rate, rewritten every batch.

The paper restrains the predictor's Lipschitz constant by giving it the
selector's rate scaled by what fraction of the input the selection kept. The
scaling is dynamic, so the checks that matter are that it follows the mask,
that it is floored, that it is written from the rate the group was built at
rather than from the rate the last batch left behind, and that an evaluation
pass does not move it.
"""

from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import SPPOutput
from pyhighlights.configurations.keys import GRU_DR

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


def rates(model, optimizer):
    predictor = {id(group) for group, _ in model.predictor_groups}
    return (
        [group["lr"] for group in optimizer.param_groups if id(group) in predictor],
        [group["lr"] for group in optimizer.param_groups if id(group) not in predictor],
    )


def test_the_predictor_trains_at_the_rate_of_what_it_reads():
    model = Registry.from_key(GRU_DR)
    optimizer = model.configure_optimizers()

    model.restrain_predictor(batch_of(), output_keeping(1))
    predictor, selector = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.25]
    assert selector == [BASE_LR]

    # Half the input the next batch: read from the base rate, not from the
    # quarter the last batch left, which would compound to a sixteenth.
    model.restrain_predictor(batch_of(), output_keeping(2))
    predictor, _ = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.5]


def test_a_selection_that_keeps_almost_nothing_hits_the_floor():
    model = Registry.from_key(GRU_DR, scale_floor=0.5)
    optimizer = model.configure_optimizers()

    model.restrain_predictor(batch_of(), output_keeping(1))
    predictor, _ = rates(model, optimizer)
    assert predictor == [BASE_LR * 0.5]


def test_both_halves_of_a_split_predictor_group_are_scaled():
    """``encoder_lr`` splits the predictor in two; the scale applies to both."""
    model = Registry.from_key(GRU_DR, encoder_lr=2e-5)
    optimizer = model.configure_optimizers()
    assert len(model.predictor_groups) == 2

    model.restrain_predictor(batch_of(), output_keeping(1))
    predictor, selector = rates(model, optimizer)
    assert sorted(predictor) == [2e-5 * 0.25, BASE_LR * 0.25]
    assert sorted(selector) == [2e-5, BASE_LR]


def test_an_untrained_model_says_so_rather_than_training_at_one_rate():
    model = Registry.from_key(GRU_DR)
    with pytest.raises(RuntimeError):
        model.restrain_predictor(batch_of(), output_keeping(1))


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

    assert [group["lr"] for group in optimizer.param_groups] == before


def test_the_rate_is_written_inside_a_real_training_loop(tmp_path):
    """The hook has to fire while Lightning owns the step, not only when called.

    ``record`` runs before the optimizer steps, which is the whole reason the
    rescale lives there. A unit call cannot show that ordering; a fit can.
    """
    import lightning as L

    from pyhighlights.components.tasks import SPPTask
    from pyhighlights.configurations.keys import TOY

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
        limit_train_batches=2,
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
