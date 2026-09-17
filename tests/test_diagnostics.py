"""Diagnostics: the stages report themselves, and only when asked.

Two things are worth a test here. What the record says, since a line nobody
can read answers nothing; and what it costs when it is off, since every stage
calls into this on every batch.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.preprocessors import LengthFilter, Pipeline
from pyhighlights.configurations.keys import TOY_TASK
from pyhighlights.utility import diagnostics


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


@pytest.fixture(autouse=True)
def quiet():
    """Leave the logger as the suite found it, whatever a test did to it."""
    level, handlers = diagnostics.logger.level, list(diagnostics.logger.handlers)
    yield
    diagnostics.logger.setLevel(level)
    diagnostics.logger.handlers = handlers


class Poisoned:
    """Something `describe` would refuse. Nothing should reach it when off."""

    def __repr__(self):
        raise AssertionError("a stage was described while diagnostics were off")


def test_nothing_is_described_while_nothing_is_listening():
    assert not diagnostics.active()
    diagnostics.record("stage", value=Poisoned())


def test_a_tensor_reports_the_things_a_metric_hides():
    described = diagnostics.describe(th.tensor([[1.0, float("nan"), 3.0]]))

    assert "tensor(1, 3)" in described
    assert "torch.float32" in described
    assert "non-finite=1" in described
    # The range covers the finite entries, since a `nan` has no place in one.
    assert "range=[1, 3]" in described


def test_an_empty_tensor_has_no_range_and_says_so():
    assert "range=[]" in diagnostics.describe(th.zeros(0))


def test_a_frame_reports_its_rows_and_how_many_carry_an_annotation():
    frame = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "tokens": [["a"], ["b"]],
            "label": [0, 1],
            "highlights": [[1], None],
        }
    )

    described = diagnostics.describe(frame)

    assert "rows=2" in described
    assert "annotated=1" in described


def test_a_long_value_is_shortened_rather_than_written_whole():
    assert diagnostics.describe("x" * 500).endswith("...")


def test_a_value_named_like_the_stage_does_not_collide_with_it(caplog):
    """The values are named by whatever the caller reports.

    Split names reach `record` as keywords, so a corpus with a split called
    `stage` would otherwise raise inside the call that exists to explain the
    run.
    """
    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        diagnostics.record("loader", **{"stage": th.zeros(1), "train": th.zeros(2)})

    assert "loader: stage = tensor(1,)" in caplog.text
    assert "loader: train = tensor(2,)" in caplog.text


def test_the_root_logger_does_not_turn_every_stage_on():
    """`logging.basicConfig(level=DEBUG)` is how a caller sees this library's
    own progress messages, and it used to enable every stage of every run --
    past the bound a task refuses a diagnosed run without.
    """
    root = logging.getLogger()
    level = root.level
    try:
        root.setLevel(logging.DEBUG)
        assert not diagnostics.active()
    finally:
        root.setLevel(level)


def test_a_loss_term_named_like_a_field_of_the_record_does_not_collide(caplog):
    """Term names are registered by a study, so they are not this module's to
    choose. `total` and `namespace` are fields the record itself writes.
    """
    from pyhighlights.configurations.keys import CROSS_ENTROPY
    from pyhighlights.utility.losses import Loss, compute_losses

    values = {"class_logits": th.randn(2, 2), "y_true": th.tensor([0, 1])}
    terms = [
        Loss(name=name, loss=CROSS_ENTROPY, inputs=["class_logits", "y_true"])
        for name in ("total", "namespace")
    ]

    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        total, computed = compute_losses(terms, values)

    assert set(computed) == {"total", "namespace"}
    assert th.isfinite(total)


def test_the_namespace_is_reported_even_when_a_binding_finds_nothing(caplog):
    """The case the record exists for: a binding whose field is absent raises,
    and the namespace is what names the fields that were there.
    """
    from pyhighlights.configurations.keys import CROSS_ENTROPY
    from pyhighlights.utility.losses import Loss, compute_losses

    missing = Loss(name="x", loss=CROSS_ENTROPY, inputs=["nowhere", "y_true"])
    values = {"class_logits": th.randn(2, 2), "y_true": th.tensor([0, 1])}

    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        with pytest.raises(KeyError, match="misses input fields"):
            compute_losses([missing], values)

    assert "loss: namespace = ['class_logits', 'y_true']" in caplog.text


def test_a_bound_that_bounds_nothing_is_refused():
    """A validation bound leaves training unbounded, and Lightning reads a
    float as a fraction, so `limit_train_batches=1.0` is its own default.
    """
    for unbounded in ({"limit_val_batches": 2}, {"limit_train_batches": 1.0}):
        with pytest.raises(ValueError, match="smoke test"):
            Registry.from_key(TOY_TASK, diagnostics=True, trainer_args=unbounded)

    for bounded in (
        {"limit_train_batches": 2},
        {"limit_train_batches": 0.1},
        {"overfit_batches": 2},
    ):
        Registry.from_key(TOY_TASK, diagnostics=True, trainer_args=bounded)


def test_a_search_is_bounded_by_the_search_rather_than_by_the_trainer():
    """`trainer_args` reaches only the trainer each candidate's predictor is
    fitted with. The search reads the whole split once per candidate.
    """
    from pyhighlights.configurations.keys import GRU_GENSPP_TRAINER, TOY_GENSPP_TASK

    Registry.from_key(
        TOY_GENSPP_TASK, diagnostics=True, trainer_args={"fast_dev_run": True}
    )
    with pytest.raises(ValueError, match="5050 candidates"):
        Registry.from_key(
            TOY_GENSPP_TASK,
            diagnostics=True,
            trainer_args={"fast_dev_run": True},
            search=GRU_GENSPP_TRAINER,
        )


def test_nothing_is_reduced_while_nothing_is_listening():
    """Two stages hold a tensor they would have to reduce to report.

    Counting the repaired rows is a reduction and an `int()` on it is a device
    synchronisation, so a run that never asked for diagnostics used to pay one
    per forward pass.
    """
    from pyhighlights.configurations.keys import GRU_FR

    model = Registry.from_key(GRU_FR)
    batch = InputData(
        features=th.randint(1, 8, (2, 5)),
        mask=th.ones((2, 5)),
        sample_ids=th.arange(2),
        y_true=th.randint(0, 2, (2,)),
        highlight_true=th.full((2, 5), -1),
    )
    calls = []
    original = diagnostics.describe
    diagnostics.describe = lambda value: calls.append(value) or original(value)
    try:
        model(batch)
    finally:
        diagnostics.describe = original

    assert not calls


def test_writing_sends_the_record_to_the_run_and_then_stops(tmp_path):
    with diagnostics.writing(tmp_path) as path:
        assert diagnostics.active()
        diagnostics.record("stage", value=th.zeros(2))

    assert "stage: value" in path.read_text()
    # And the logger is as it was, so a second task in this process does not
    # keep writing into the first one's directory.
    assert not diagnostics.active()
    assert not diagnostics.logger.handlers


def test_every_preprocessing_step_reports_what_it_returned(caplog):
    splits = {
        "train": pd.DataFrame(
            {
                "sample_id": [0, 1],
                "text": ["a b", "c"],
                "tokens": [["a", "b"], ["c"]],
                "label": [0, 1],
                "highlights": [None, None],
            }
        )
    }
    pipeline = Pipeline()
    pipeline.preprocessors = [LengthFilter(max_length=1)]

    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        pipeline.process(splits)

    assert "preprocessor.LengthFilter: train = frame rows=1" in caplog.text


def test_a_run_nobody_bounded_is_refused():
    with pytest.raises(ValueError, match="smoke test"):
        Registry.from_key(TOY_TASK, diagnostics=True)

    # A bound of either kind is enough, and both are Lightning's own.
    Registry.from_key(TOY_TASK, diagnostics=True, trainer_args={"fast_dev_run": True})
    Registry.from_key(
        TOY_TASK, diagnostics=True, trainer_args={"limit_train_batches": 2}
    )


def test_a_bounded_run_records_every_stage_beside_its_results(tmp_path):
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        diagnostics=True,
        trainer_args={"fast_dev_run": True},
    )

    task.run()

    written = (task.directory / diagnostics.FILENAME).read_text()
    for stage in (
        "step",
        "loader",
        "collator",
        "selector",
        "repair",
        "predictor",
        "loss",
        "metric",
    ):
        assert f"{stage}:" in written, stage
    # The two axes, side by side, which is where an alignment error shows.
    assert "collator: subtokens" in written and "collator: words" in written
    # And the namespace every binding reads, so a missing field has a name.
    assert "loss: namespace" in written
    assert not diagnostics.active()


def test_a_phased_model_says_which_phase_a_line_belongs_to(tmp_path, caplog):
    """Without this a record is undifferentiated tensors.

    A phased model passes each batch through every stage twice, on two
    different selections scored by two different criteria, and it drives its
    own optimizers -- so it never reaches `Model._step` and has to mark the
    split itself. A marker that goes missing does not fail: the lines simply
    read as belonging to whatever ran before them.
    """
    from pyhighlights.configurations.keys import GRU_MCD

    model = Registry.from_key(GRU_MCD)
    batch = InputData(
        features=th.randint(1, 8, (2, 5)),
        mask=th.ones((2, 5)),
        sample_ids=th.arange(2),
        y_true=th.randint(0, 2, (2,)),
        highlight_true=th.full((2, 5), -1),
    )

    with caplog.at_level(logging.DEBUG, logger=diagnostics.logger.name):
        model.predictor_phase_loss(batch)
        model.generator_phase_loss(batch)

    assert "phase: name = 'classifier'" in caplog.text
    assert "phase: name = 'generator'" in caplog.text


def test_a_run_that_was_not_asked_writes_nothing(tmp_path):
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        trainer_args={"fast_dev_run": True},
    )

    task.run()

    assert not (task.directory / diagnostics.FILENAME).exists()


def test_the_metric_bindings_are_said_once_a_split_not_once_a_batch(tmp_path):
    """They are a property of the registration, not of the batch.

    The namespace a metric binds to and the fields it names are the same on
    every step of a split, so repeating them per batch buries the lines that
    do change underneath them.
    """
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        diagnostics=True,
        trainer_args={"limit_train_batches": 2, "max_epochs": 2},
    )

    task.run()

    written = (task.directory / diagnostics.FILENAME).read_text()
    assert written.count("metric: split = 'train'") == 1
    assert written.count("step: split = 'train'") > 1
