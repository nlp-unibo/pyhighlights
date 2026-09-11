"""DAR: a second predictor that only ever read full text scores the highlight.

The method is that module and the fact that it is frozen. So what the tests
are about is that it is separate from the predictor, that it is trained before
the first rationalization epoch and not after, that its term reaches the
generator and nothing else, and that a resumed run finds it trained rather
than training it again on a model that has already moved.
"""

from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import DAR
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import (
    ALIGNMENT_CLASSIFICATION_LOSS,
    GRU_DAR,
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


def toy_loader(tmp_path):
    task = SPPTask(
        loader=TOY,
        model=GRU_DAR,
        save_path=str(tmp_path),
        batch_size=8,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    return task.loaders(task.splits())["train"]


def fit(model, loader) -> None:
    L.Trainer(
        accelerator="cpu",
        max_epochs=1,
        limit_train_batches=2,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    ).fit(model, loader)


def trained(tmp_path) -> DAR:
    """One short fit over the toy corpus, pretraining included."""
    model = Registry.from_key(GRU_DAR, pretrain_epochs=1)
    fit(model, toy_loader(tmp_path))
    return model


def test_the_alignment_term_reads_the_aligner_not_the_predictor():
    loss = Registry.from_key(ALIGNMENT_CLASSIFICATION_LOSS)

    assert loss.inputs == ["aligner_class_logits", "y_true"]
    assert loss.coefficient > 0


def test_the_aligner_is_a_module_of_its_own():
    model = Registry.from_key(GRU_DAR)
    predictor = {id(parameter) for parameter in model.predictor.parameters()} | {
        id(parameter) for parameter in model.predictor_backbone.parameters()
    }

    assert not predictor & {id(p) for p in model.aligner_parameters()}
    # And the model carries the term whether or not a registration listed it.
    assert "alignment_classification" in {loss.name for loss in model.losses}


def test_the_optimizer_leaves_the_aligner_out():
    model = Registry.from_key(GRU_DAR)
    optimizer = model.configure_optimizers()
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert not optimized & {id(p) for p in model.aligner_parameters()}
    assert {id(p) for p in model.predictor.parameters()} <= optimized


def test_the_encoder_rate_reaches_the_aligner_encoder():
    """`encoder_lr` is about where a parameter sits, and the aligner has one."""
    model = Registry.from_key(GRU_DAR, encoder_lr=2e-5)
    encoders = model.encoder_ids()

    assert {id(p) for p in model.aligner_backbone.parameters()} <= encoders


def test_pretrained_embeddings_reach_the_aligner():
    model = Registry.from_key(GRU_DAR)
    table = model.aligner_backbone.embedding.weight
    matrix = th.full_like(table, 0.25)

    model.load_embeddings(matrix)

    assert th.equal(model.aligner_backbone.embedding.weight, matrix)


def test_an_aligner_that_never_read_anything_is_refused():
    with pytest.raises(ValueError, match="pretraining epoch"):
        Registry.from_key(GRU_DAR, pretrain_epochs=0)


def test_the_aligner_is_trained_before_the_first_epoch_and_frozen_after(tmp_path):
    model = Registry.from_key(GRU_DAR, pretrain_epochs=1)
    loader = toy_loader(tmp_path)
    before = [parameter.detach().clone() for parameter in model.aligner_parameters()]

    fit(model, loader)

    assert bool(model.aligner_ready)
    assert any(
        not th.equal(was, now) for was, now in zip(before, model.aligner_parameters())
    )
    # And nothing moves it once the rationalizer starts.
    assert not any(parameter.requires_grad for parameter in model.aligner_parameters())
    after = [parameter.detach().clone() for parameter in model.aligner_parameters()]

    fit(model, loader)

    assert all(
        th.equal(was, now) for was, now in zip(after, model.aligner_parameters())
    )


def test_a_resumed_model_keeps_the_aligner_it_was_checkpointed_with(tmp_path):
    model = trained(tmp_path)
    state = model.state_dict()

    resumed = Registry.from_key(GRU_DAR, pretrain_epochs=1)
    resumed.load_state_dict(state)
    assert bool(resumed.aligner_ready)

    aligner = [parameter.detach().clone() for parameter in resumed.aligner_parameters()]
    fit(resumed, toy_loader(tmp_path))

    assert all(
        th.equal(was, now) for was, now in zip(aligner, resumed.aligner_parameters())
    )


def test_the_alignment_term_trains_the_generator_alone(tmp_path):
    """Frozen aligner, and no path from this term to the predictor.

    The aligner reads the highlight, so the only trainable thing behind its
    logits is the module that chose it.
    """
    model = trained(tmp_path)
    model.zero_grad(set_to_none=True)
    batch = batch_of()

    output = model(batch)
    highlight = model.aggregator(output).highlight_mask
    model.align(batch, highlight).sum().backward()

    assert model.selectors[0].selector[-1].weight.grad is not None
    assert model.predictor.predictor[-1].weight.grad is None
    assert all(parameter.grad is None for parameter in model.aligner_parameters())


def test_the_aligner_reads_the_highlight_and_not_the_rest(tmp_path):
    model = trained(tmp_path)
    model.eval()
    batch = batch_of()
    width = batch.mask.shape[1]

    keep_first = th.zeros((batch.mask.shape[0], width))
    keep_first[:, 0] = 1.0
    keep_last = th.zeros_like(keep_first)
    keep_last[:, -1] = 1.0

    with th.no_grad():
        assert not th.allclose(
            model.align(batch, keep_first), model.align(batch, keep_last)
        )
        # And the full input is what it was taught on, so that pass is the
        # same computation with nothing dropped.
        assert th.allclose(
            model.align_full(batch), model.align(batch, th.ones_like(keep_first))
        )
