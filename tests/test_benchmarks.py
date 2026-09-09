import json
from pathlib import Path

import pandas as pd
import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.analyzers import (
    HighlightPositionAnalyzer,
    MetricsAnalyzer,
    PredictionAnalyzer,
    latex_table,
)
from pyhighlights.components.benchmarks import Benchmark
from pyhighlights.components.tasks import Task
from pyhighlights.configurations.keys import (
    HIGHLIGHT_POSITION_ANALYZER,
    METRICS_ANALYZER,
    PREDICTION_ANALYZER,
    TOY,
    TOY_BENCHMARK,
    TOY_TASK,
)


class FakeTask(Task):
    """A task that reports without training anything."""

    def __init__(self, summary: dict, fails: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.summary = summary
        self.fails = fails

    def run(self):
        if self.fails:
            raise RuntimeError("this configuration cannot run")
        results = {"name": self.name, "seeds": [0], "runs": [], "summary": self.summary}
        self.serialize(results)
        return results


def summary(**metrics) -> dict:
    return {
        f"test_{name}": {"mean": value, "std": 0.1, "values": [value]}
        for name, value in metrics.items()
    }


def test_a_benchmark_runs_its_tasks_and_collects_them(tmp_path, monkeypatch):
    tasks = {
        "first": FakeTask(summary(accuracy=0.8), name="first"),
        "second": FakeTask(summary(accuracy=0.6), name="second"),
    }
    benchmark = Benchmark(tasks=list(tasks), name="grid", save_path=str(tmp_path))
    monkeypatch.setattr(
        benchmark,
        "build",
        lambda key: tasks[key].__class__(
            tasks[key].summary,
            name=tasks[key].name,
            save_path=str(benchmark.directory),
        ),
    )

    report = benchmark.run()

    assert [item["task"] for item in report["tasks"]] == ["first", "second"]
    assert report["failed"] == []
    written = json.loads((tmp_path / "grid" / "benchmark.json").read_text())
    assert written == report
    # Each task wrote inside the benchmark's own directory, in a run of its own.
    assert list((tmp_path / "grid" / "first").glob("*/results.json"))


def test_a_failing_task_does_not_take_the_grid_with_it(tmp_path, monkeypatch):
    tasks = {
        "broken": FakeTask(summary(), fails=True, name="broken"),
        "fine": FakeTask(summary(accuracy=0.9), name="fine"),
    }
    benchmark = Benchmark(tasks=list(tasks), name="grid", save_path=str(tmp_path))
    monkeypatch.setattr(benchmark, "build", lambda key: tasks[key])

    report = benchmark.run()

    assert report["failed"] == ["broken"]
    assert "cannot run" in report["tasks"][0]["error"]
    assert report["tasks"][1]["summary"]["test_accuracy"]["mean"] == 0.9

    strict = Benchmark(
        tasks=["broken"], name="strict", save_path=str(tmp_path), strict=True
    )
    monkeypatch.setattr(strict, "build", lambda key: tasks[key])
    with pytest.raises(RuntimeError, match="cannot run"):
        strict.run()


def test_a_benchmark_needs_a_task():
    with pytest.raises(ValueError, match="at least one task"):
        Benchmark(tasks=[])


def test_metrics_analyzer_reads_every_result_beneath_a_directory(tmp_path):
    for name, accuracy in (("fr", 0.8), ("mgr", 0.75)):
        FakeTask(
            summary(accuracy=accuracy, highlight_f1=0.4),
            name=name,
            save_path=str(tmp_path),
        ).run()
    # A task that measured something the others did not.
    FakeTask(summary(accuracy=0.5), name="toy", save_path=str(tmp_path)).run()

    report = MetricsAnalyzer(
        directory=tmp_path, metrics=["accuracy", "highlight_f1"]
    ).analyze()

    assert list(report.columns) == ["task", "run", "seeds", "accuracy", "highlight_f1"]
    assert sorted(report["task"]) == ["fr", "mgr", "toy"]
    assert report.set_index("task").loc["fr", "accuracy"] == "0.8000 +/- 0.1000"
    # A metric a task never measured reads as absent, not as zero.
    assert report.set_index("task").loc["toy", "highlight_f1"] == "-"

    every = MetricsAnalyzer(directory=tmp_path).analyze()
    assert "accuracy" in every.columns

    pairs = MetricsAnalyzer(
        directory=tmp_path, metrics=["accuracy"], pairs=True
    ).analyze()
    assert pairs.set_index("task").loc["fr", "accuracy"] == (0.8, 0.1)
    assert r"_{\pm 10.00}" in latex_table(pairs[["accuracy"]])


def test_metrics_analyzer_reports_nothing_for_an_empty_directory(tmp_path):
    assert MetricsAnalyzer(directory=tmp_path).analyze().empty


def test_highlight_position_analyzer_bins_where_the_selector_looked(tmp_path):
    run = tmp_path / "2026-01-01T00-00-00"
    run.mkdir()
    batch = {
        # Two documents: the first selects its opening token, the second its
        # closing one, and the padded position is not a document position.
        "highlight_mask": [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
        "mask": [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.0]],
    }
    pd.to_pickle([batch], run / "predictions-seed=0.pkl")

    report = HighlightPositionAnalyzer(directory=tmp_path, bins=2).analyze()

    assert report.loc[0, "run"] == "2026-01-01T00-00-00"
    assert report.loc[0, "seed"] == "0"
    assert report.loc[0, "samples"] == 2
    assert report.loc[0, "selection_rate"] == pytest.approx(2 / 7)
    assert report.loc[0, "bin_0"] == pytest.approx(0.5)
    assert report.loc[0, "bin_1"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="bins must be positive"):
        HighlightPositionAnalyzer(bins=0)


def test_an_empty_document_is_skipped(tmp_path):
    run = tmp_path / "2026-01-01T00-00-00"
    run.mkdir()
    pd.to_pickle(
        [{"highlight_mask": [[0.0, 0.0]], "mask": [[0.0, 0.0]]}],
        run / "predictions-seed=0.pkl",
    )

    report = HighlightPositionAnalyzer(directory=tmp_path, bins=2).analyze()
    assert report.loc[0, "samples"] == 0
    assert report.loc[0, "selection_rate"] == 0.0
    assert report.loc[0, "bin_0"] == 0.0


def test_registered_benchmark_and_analyzers_build(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    benchmark = Registry.from_key(TOY_BENCHMARK, save_path=str(tmp_path))
    assert isinstance(benchmark, Benchmark)
    assert benchmark.tasks == [TOY_TASK]
    # The task it builds is told to save inside the benchmark.
    assert benchmark.build(TOY_TASK).save_path == benchmark.directory

    assert isinstance(Registry.from_key(METRICS_ANALYZER), MetricsAnalyzer)
    assert isinstance(
        Registry.from_key(HIGHLIGHT_POSITION_ANALYZER), HighlightPositionAnalyzer
    )


def test_run_prints_the_report_it_returns(tmp_path, capsys):
    """``run`` is what ``cmn-run`` calls, so the report has to reach a terminal."""
    FakeTask(summary(accuracy=0.8), name="fr", save_path=str(tmp_path)).run()

    report = MetricsAnalyzer(directory=tmp_path, metrics=["accuracy"]).run()

    assert "0.8000 +/- 0.1000" in capsys.readouterr().out
    assert not report.empty


def test_latex_table_typesets_pairs_and_escapes_the_rest():
    frame = pd.DataFrame({"model": ["gru_fr"], "highlight_f1": [(0.1234, 0.0056)]})

    table = latex_table(frame)
    header, row = table.splitlines()

    # An unescaped underscore is a subscript, and every metric name has one.
    assert header == r"model & highlight\_f1 \\"
    assert row == r"gru\_fr & $12.34_{\pm 0.56}$ \\"
    assert latex_table(frame, percentage=False).endswith(r"$0.12_{\pm 0.01}$ \\")


def test_position_analyzer_reads_the_head_the_metrics_score(tmp_path):
    """A multi-head model stores one mask per head; the aggregator keeps one."""
    run = tmp_path / "2026-01-01T00-00-00"
    run.mkdir()
    batch = {
        # Head 0 selects the opening token, the other heads the closing one.
        "highlight_mask": [
            [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]
        ],
        "mask": [[1.0, 1.0, 1.0, 1.0]],
    }
    pd.to_pickle([batch], run / "predictions-seed=0.pkl")

    report = HighlightPositionAnalyzer(directory=tmp_path, bins=2).analyze()

    assert report.loc[0, "samples"] == 1
    assert report.loc[0, "selection_rate"] == pytest.approx(0.25)
    assert report.loc[0, "bin_0"] == pytest.approx(1.0)
    assert report.loc[0, "bin_1"] == pytest.approx(0.0)


def test_a_registered_benchmark_runs_the_task_it_names(tmp_path):
    """No monkeypatching: the key is built, the override lands, the task runs."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    benchmark = Registry.from_key(TOY_BENCHMARK, save_path=str(tmp_path))

    report = benchmark.run()

    assert report["failed"] == []
    assert [item["task"] for item in report["tasks"]] == ["toy"]
    assert list((benchmark.directory / "toy").glob("*/results.json"))
    assert (benchmark.directory / "benchmark.json").exists()
    # The report the benchmark wrote is JSON, so every number in it survived.
    assert json.loads((benchmark.directory / "benchmark.json").read_text()) == report


def test_the_prediction_analyzer_reports_what_the_selector_kept(tmp_path):
    """End to end: a real run's predictions, joined back to its own corpus."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        store_predictions=True,
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )
    task.run()

    report = PredictionAnalyzer(directory=tmp_path).analyze()

    assert not report.empty
    assert list(report.columns) == [
        "run",
        "seed",
        "sample_id",
        "label",
        "predicted",
        "tokens",
        "selected",
        "rationale",
        "highlights",
    ]
    # One row per test sample, and the words are words rather than ids.
    corpus = Registry.from_key(TOY).load()["test"]
    assert sorted(report["sample_id"]) == sorted(corpus["sample_id"])
    assert report["seed"].unique().tolist() == ["42"]
    for row in report.itertuples(index=False):
        assert all(isinstance(token, str) for token in row.tokens)
        # A selection names positions in the words it selected from, and the
        # rationale is those words.
        assert all(0 <= word < len(row.tokens) for word in row.selected)
        assert row.rationale == " ".join(row.tokens[word] for word in row.selected)
        assert row.label in (0, 1)
        assert row.predicted in (0, 1)


def write_run(directory: Path, batch: dict) -> Path:
    """A run directory holding one seed's predictions and its manifest."""
    run = directory / "2026-01-01T00-00-00"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps({"settings": {"loader": {"key": str(TOY)}, "preprocessor": None}})
    )
    pd.to_pickle([batch], run / "predictions-seed=7.pkl")
    return run


def test_a_selected_subtoken_selects_its_whole_word(tmp_path):
    """Selections are made over subtokens and reported over words."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    sample_id = int(Registry.from_key(TOY).load()["test"]["sample_id"].iloc[0])
    write_run(
        tmp_path,
        {
            # Four positions over two words, then a padded one. The second
            # subtoken of word 1 is selected and nothing else is.
            "word_ids": [[0, 0, 1, 1, -1]],
            "mask": [[1.0, 1.0, 1.0, 1.0, 0.0]],
            "highlight_mask": [[0.0, 0.0, 0.0, 1.0, 1.0]],
            "class_logits": [[0.1, 0.9]],
            "sample_ids": [sample_id],
        },
    )

    report = PredictionAnalyzer(directory=tmp_path).analyze()

    (row,) = report.itertuples(index=False)
    assert row.run == "2026-01-01T00-00-00"
    assert row.seed == "7"
    assert row.sample_id == sample_id
    # Word 1 selected once, word 0 never, and the padded position is nobody's.
    assert row.selected == [1]
    assert row.predicted == 1


def test_a_sample_the_corpus_no_longer_holds_is_skipped(tmp_path):
    """A corpus that changed under a run still reports the samples it kept."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    write_run(
        tmp_path,
        {
            "word_ids": [[0, -1]],
            "mask": [[1.0, 0.0]],
            "highlight_mask": [[1.0, 0.0]],
            "class_logits": [[0.9, 0.1]],
            "sample_ids": [10_000],
        },
    )

    assert PredictionAnalyzer(directory=tmp_path).analyze().empty


def test_the_prediction_analyzer_is_registered():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    assert isinstance(Registry.from_key(PREDICTION_ANALYZER), PredictionAnalyzer)


def test_the_prediction_analyzer_reads_the_head_the_metrics_score(tmp_path):
    """``forward`` stacks heads on axis 1, so ``[:, 0]`` is the first head."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    sample_id = int(Registry.from_key(TOY).load()["test"]["sample_id"].iloc[0])
    write_run(
        tmp_path,
        {
            "word_ids": [[0, 1]],
            "mask": [[1.0, 1.0]],
            # Head 0 selects the first word and calls it class 0; the other
            # head selects the second and calls it class 1.
            "highlight_mask": [[[1.0, 0.0], [0.0, 1.0]]],
            "class_logits": [[[0.9, 0.1], [0.1, 0.9]]],
            "sample_ids": [sample_id],
        },
    )

    (row,) = PredictionAnalyzer(directory=tmp_path).analyze().itertuples(index=False)

    assert row.selected == [0]
    assert row.predicted == 0


def test_a_word_the_corpus_no_longer_has_skips_the_sample(tmp_path):
    """A corpus that placed its words differently cannot read the selection."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    sample_id = int(Registry.from_key(TOY).load()["test"]["sample_id"].iloc[0])
    write_run(
        tmp_path,
        {
            "word_ids": [[0, 9_999]],
            "mask": [[1.0, 1.0]],
            "highlight_mask": [[1.0, 1.0]],
            "class_logits": [[0.9, 0.1]],
            "sample_ids": [sample_id],
        },
    )

    assert PredictionAnalyzer(directory=tmp_path).analyze().empty


def test_a_padded_position_is_nobody_s_word(tmp_path):
    """``-1`` marks a position with no source word and never reaches a row."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    sample_id = int(Registry.from_key(TOY).load()["test"]["sample_id"].iloc[0])
    write_run(
        tmp_path,
        {
            # A mask that wrongly calls the padded position valid, and a
            # selector that selects it.
            "word_ids": [[0, -1]],
            "mask": [[1.0, 1.0]],
            "highlight_mask": [[0.0, 1.0]],
            "class_logits": [[0.9, 0.1]],
            "sample_ids": [sample_id],
        },
    )

    (row,) = PredictionAnalyzer(directory=tmp_path).analyze().itertuples(index=False)

    assert row.selected == []
    assert row.rationale == ""
