"""What a run cost: the columns, and what makes them comparable across rows.

The arithmetic is the point here. A wall clock is a wall clock; what a test
can pin is that a search which trained five thousand models eight at a time
reports what one of them cost, and that a model trained by descent reports its
own wall clock through the same columns.
"""

import resource
from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.analyzers import MetricsAnalyzer
from pyhighlights.components.tasks import GenSPPTask
from pyhighlights.configurations.keys import (
    TOY_GENSPP_TASK,
    TOY_GENSPP_TRAINER,
    TOY_TASK,
)
from pyhighlights.utility import cost

SMOKE = {
    "accelerator": "cpu",
    "max_epochs": 1,
    "enable_progress_bar": False,
    "enable_model_summary": False,
}


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def test_a_frozen_parameter_is_counted_and_told_apart():
    """It is memory and compute at inference however little it learns.

    And a model that freezes most of itself is a different proposition to
    train than one that does not, so the two are separate columns.
    """
    model = th.nn.Linear(4, 2)
    model.bias.requires_grad_(False)

    assert cost.parameters(model) == 10
    assert cost.parameters(model, trainable=True) == 8
    assert cost.parameters(model, trainable=False) == 2


def test_a_search_reports_what_one_of_its_candidates_cost():
    """Wall clock alone says a search is as cheap as the machine made it.

    Eight workers do eight seconds of work in a second, so the product is
    worker-seconds and the models trained divide it.
    """
    meter = cost.Meter(concurrency=8, models=5050)
    meter.runtime = 3600.0
    meter.peak = 8000.0

    columns = meter.columns(th.nn.Linear(4, 2))

    assert columns["cost_runtime_s"] == 3600.0
    assert columns["cost_runtime_per_run_s"] == pytest.approx(3600.0 * 8 / 5050)
    # Counts, so they stay integers: `5050.0` models reads as a measurement.
    assert (columns["cost_concurrency"], columns["cost_models"]) == (8, 5050)


def test_one_model_on_one_worker_reports_its_own_wall_clock():
    """Which is what makes the per-run column comparable at all."""
    meter = cost.Meter()
    meter.runtime, meter.peak = 12.0, 500.0

    columns = meter.columns(th.nn.Linear(4, 2))

    assert columns["cost_runtime_per_run_s"] == 12.0


def test_a_run_trains_at_least_one_model_on_at_least_one_worker():
    for impossible in ({"concurrency": 0}, {"models": 0}):
        with pytest.raises(ValueError, match="at least one"):
            cost.Meter(**impossible)


def test_a_timer_that_saw_no_test_pass_reports_nothing():
    """Rather than a zero, which reads as an inference that took no time."""
    assert cost.InferenceTimer().columns() == {}


def test_the_timer_reports_the_pass_and_its_batches():
    timer = cost.InferenceTimer()
    timer.on_test_epoch_start(None, None)
    for _ in range(4):
        timer.on_test_batch_start()
        timer.on_test_batch_end()
    timer.on_test_epoch_end(None, None)

    columns = timer.columns()

    assert columns["cost_inference_epoch_s"] >= columns["cost_inference_batch_s"]
    assert columns["cost_inference_batch_s"] > 0


def test_a_seed_reports_what_it_cost_beside_what_it_scored(tmp_path):
    task = Registry.from_key(
        TOY_TASK, save_path=str(tmp_path), seeds=[0], trainer_args=SMOKE
    )

    results = task.run()

    run = results["runs"][0]
    assert run["cost_runtime_s"] > 0
    assert run["cost_inference_batch_s"] > 0
    assert run["cost_memory_mib"] > 0
    assert run["cost_parameters"] > 0
    # Split, and the two halves are the whole.
    assert (
        run["cost_trainable_parameters"] + run["cost_frozen_parameters"]
        == run["cost_parameters"]
    )
    # A model trained by descent is one model, on one worker.
    assert (run["cost_concurrency"], run["cost_models"]) == (1, 1)
    assert run["cost_runtime_per_run_s"] == run["cost_runtime_s"]
    # And they summarise like any other column, so seeds can be compared.
    assert results["summary"]["cost_runtime_s"]["values"] == [run["cost_runtime_s"]]


def test_a_search_counts_the_candidates_it_actually_trained(tmp_path):
    """The founders plus every generation's children, and the workers.

    Budgeted rather than run would overcount a search that reached
    `stop_threshold` early, which is the one a study most wants costed.
    """
    task = Registry.from_key(
        TOY_GENSPP_TASK, save_path=str(tmp_path), seeds=[0], trainer_args=SMOKE
    )
    search = Registry.from_key(TOY_GENSPP_TRAINER)

    run = task.run()["runs"][0]

    assert run["cost_models"] == GenSPPTask.candidates(search)
    assert run["cost_concurrency"] == len(search.devices)
    assert run["cost_runtime_per_run_s"] == pytest.approx(
        run["cost_runtime_s"] * run["cost_concurrency"] / run["cost_models"]
    )


def test_the_cost_table_is_the_metrics_table_with_another_prefix(tmp_path):
    """So a computational comparison is a call, not a second analyzer."""
    Registry.from_key(
        TOY_TASK, save_path=str(tmp_path), seeds=[0], trainer_args=SMOKE
    ).run()

    table = MetricsAnalyzer(directory=tmp_path, split="cost").analyze()

    assert "runtime_s" in table.columns and "parameters" in table.columns
    # And the table a paper quotes carries none of them.
    scores = MetricsAnalyzer(directory=tmp_path, split="test").analyze()
    assert not [name for name in scores.columns if name.startswith("cost")]


def test_a_meter_is_a_context_manager_too():
    """For a run measured in one place rather than started and stopped."""
    with cost.Meter() as meter:
        sum(range(10000))

    assert meter.runtime > 0 and meter.peak > 0


def test_a_pool_wider_than_the_population_is_not_the_concurrency(tmp_path):
    """Eight workers cannot run four candidates eight at a time.

    Counting them as eight would report each candidate as costing twice what
    it did, which is the smoke-sized search every wiring check runs.
    """
    task = Registry.from_key(
        TOY_GENSPP_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        trainer_args=SMOKE,
    )

    run = task.run()["runs"][0]

    assert run["cost_concurrency"] <= run["cost_models"]
    assert run["cost_runtime_per_run_s"] <= run["cost_runtime_s"]


def test_memory_is_reported_in_mebibytes():
    """The unit `nvidia-smi` and every process monitor print.

    CUDA counts bytes and `getrusage` counts kibibytes, so a run used to
    report decimal megabytes on one device and mebibytes on the other -- a
    five percent difference nothing in the table would have explained.
    """
    peak = cost.peak_memory()
    # Children included: a search scores its candidates in processes of their
    # own, and this process alone would report the parent waiting on them.
    resident = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    )

    if not th.cuda.is_available():
        assert peak == pytest.approx(resident / 1024, rel=1e-6)
    assert cost.MIB == 1048576


def test_the_peak_is_not_divided_among_the_workers():
    """Most of it is resident before the first candidate exists.

    The interpreter, torch and the corpus, measured at 521 MiB with nothing
    training, so a per-model share would report less memory than a run holds
    doing nothing -- whether the candidates are threads or processes. The
    column is one process's ceiling, and `cost_concurrency` is what says how
    many such processes a search ran at once.
    """
    meter = cost.Meter(concurrency=8, models=5050)
    meter.runtime, meter.peak = 3600.0, 8000.0

    columns = meter.columns(th.nn.Linear(4, 2))

    assert columns["cost_memory_mib"] == 8000.0
    assert not [name for name in columns if name.startswith("cost_memory_per")]
