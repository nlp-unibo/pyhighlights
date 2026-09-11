"""What a run wrote down about itself."""

import json
from datetime import datetime
from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components import tasks
from pyhighlights.components.analyzers import MetricsAnalyzer
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import GRU_FR, LEAKAGE_REMOVER, TOY, TOY_TASK
from pyhighlights.utility.manifest import (
    KEY_FIELD,
    PACKAGES,
    describe,
    registration_key,
    resolve,
    versions,
)


def test_versions_reports_the_interpreter_and_what_is_installed():
    found = versions()

    assert found["python"].count(".") == 2
    # Everything a run needs to compute is installed by the dev extra; an
    # optional package that is absent is left out rather than reported as None.
    assert {"pyhighlights", "cinnamon-core", "torch", "lightning"} <= set(found)
    assert set(found) <= {"python", *PACKAGES}
    assert None not in found.values()


def test_resolve_replaces_a_key_with_the_values_behind_it():
    """The point of the manifest: numbers, not the names they came from."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    resolved = resolve(GRU_FR)

    assert resolved[KEY_FIELD] == str(GRU_FR)
    # Two levels down, past the model and into the backbone it names.
    assert resolved["selector_backbones"]["hidden_size"] == 128
    # A list of keys keeps its shape and its order.
    assert [loss["name"] for loss in resolved["losses"]] == [
        "classification",
        "sparsity",
        "contiguity",
    ]
    assert resolved["losses"][1]["loss"]["threshold"] == 0.15


def test_resolve_leaves_a_plain_value_alone():
    assert resolve({"seeds": [1, 2], "name": "toy", "rate": 0.5}) == {
        "seeds": [1, 2],
        "name": "toy",
        "rate": 0.5,
    }
    assert resolve(frozenset({"b", "a"})) == ["a", "b"]


def test_describe_states_the_component_and_omits_what_the_run_built():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(TOY_TASK, seeds=[7])

    manifest = describe(task)

    assert manifest["component"] == "pyhighlights.components.tasks.SPPTask"
    assert manifest["settings"]["seeds"] == [7]
    assert manifest["settings"]["model"]["selectors"]["hidden_sizes"] == []
    # `_embedding_matrix` is what the run fitted, not what it was asked for.
    assert not any(name.startswith("_") for name in manifest["settings"])
    assert manifest["versions"] == versions()


def test_a_manifest_names_the_key_that_built_the_task_and_the_overrides():
    """The key alone rebuilds the defaults; the overrides say what was run."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(TOY_TASK, seeds=[7])

    manifest = describe(task)

    assert manifest["key"] == str(TOY_TASK)
    assert manifest["build_args"] == {"seeds": [7]}
    # Reported as a record of the build, not among the settings, where they
    # would read as something the task was configured with.
    assert not {"registration_key", "build_args"} & set(manifest["settings"])


def test_a_manifest_says_so_when_nobody_built_the_task():
    """A task constructed directly carries no annotation to report."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    manifest = describe(SPPTask(loader=TOY, model=GRU_FR))

    assert manifest["key"] is None
    assert manifest["build_args"] is None


def test_an_overridden_key_is_resolved_like_any_other():
    """A build arg holding a key is written out as the values behind it."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(TOY_TASK, model=GRU_FR)

    build_args = describe(task)["build_args"]

    assert build_args["model"][KEY_FIELD] == str(GRU_FR)
    assert "selectors" in build_args["model"]


def test_a_run_never_overwrites_an_earlier_one(tmp_path):
    """Two runs of one task are two results, not one amended."""
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path),
        seeds=[0],
        trainer_args={"accelerator": "cpu", "max_epochs": 1},
    )

    # The stamp is taken once per run, so every seed, the metrics and the
    # manifest of one run land in the same directory.
    task.run()
    first = task.directory
    assert task.directory == first

    task.run()
    second = task.directory

    assert second != first
    assert second.parent == first.parent == tmp_path / "toy"
    assert sorted(path.name for path in (tmp_path / "toy").iterdir()) == sorted(
        [first.name, second.name]
    )
    for run in (first, second):
        assert (run / "results.json").exists()
        assert (run / "manifest.json").exists()


def test_a_run_writes_down_the_settings_it_was_given(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        name="recorded",
        save_path=str(tmp_path),
        patience=3,
    )

    directory = task.serialize({"runs": []})
    written = json.loads((directory / "manifest.json").read_text())

    assert written["started"] == directory.name
    assert written["settings"]["patience"] == 3
    # The whole tree, so a reader has the numbers rather than the key names.
    assert written["settings"]["model"]["losses"][1]["loss"]["threshold"] == 0.15
    assert written["versions"]["pyhighlights"] == versions()["pyhighlights"]


@pytest.mark.parametrize("latest", [True, False])
def test_the_metrics_table_reports_the_newest_run_of_each_task(tmp_path, latest):
    """A re-run adds a directory, and by default only the last one is quoted."""
    for stamp, accuracy in (("2026-01-01T00-00-00", 0.5), ("2026-06-01T00-00-00", 0.9)):
        directory = tmp_path / "fr" / stamp
        directory.mkdir(parents=True)
        (directory / "results.json").write_text(
            json.dumps(
                {
                    "name": "fr",
                    "seeds": [0],
                    "summary": {
                        "test_accuracy": {"mean": accuracy, "std": 0.0, "values": []}
                    },
                }
            )
        )

    report = MetricsAnalyzer(
        directory=tmp_path, metrics=["accuracy"], pairs=True, latest=latest
    ).analyze()

    if latest:
        assert list(report["run"]) == ["2026-06-01T00-00-00"]
        assert report.loc[0, "accuracy"] == (0.9, 0.0)
    else:
        assert list(report["run"]) == [
            "2026-01-01T00-00-00",
            "2026-06-01T00-00-00",
        ]
        assert list(report["accuracy"]) == [(0.5, 0.0), (0.9, 0.0)]


class FrozenClock:
    """A clock that never moves, so two runs always start in the same second."""

    @staticmethod
    def now():
        return datetime(2026, 1, 1, 12, 0, 0)


def test_two_runs_inside_one_second_still_get_a_directory_each(tmp_path, monkeypatch):
    """The stamp is one second wide; the guarantee is not."""
    # Built here rather than borrowed from whichever test ran first: every
    # other test in this module builds its own, and this one only passed
    # because it followed them. Run it on a worker of its own -- which is
    # what `pytest -n` does -- and the registry it reads was never built.
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    # The clock is pinned because a real one produces this collision only by
    # luck: two runs that straddle a second boundary get two different stamps,
    # which is correct behaviour and a failing assertion below.
    monkeypatch.setattr(tasks, "datetime", FrozenClock)

    first = SPPTask(loader=TOY, model=GRU_FR, name="quick", save_path=str(tmp_path))
    first.serialize({"runs": []})
    second = SPPTask(loader=TOY, model=GRU_FR, name="quick", save_path=str(tmp_path))
    second.serialize({"runs": []})

    assert first.directory != second.directory
    assert second.directory.name.startswith(first.directory.name)
    assert len(list((tmp_path / "quick").iterdir())) == 2


def test_a_parameter_called_key_does_not_overwrite_the_registration_key():
    """Both survive, because the key is not stored under a parameter's name.

    `LeakageRemover` has a `key` parameter -- it names the column it
    deduplicates on -- so its resolved entry used to read `"key": "text"` with
    the registration key gone, and `PredictionAnalyzer.corpus` then handed
    `"text"` to `RegistrationKey.parse`.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    resolved = resolve(LEAKAGE_REMOVER)

    assert resolved[KEY_FIELD] == str(LEAKAGE_REMOVER)
    assert resolved["key"] == "text"
    # And a reader gets the registration key without knowing which it is.
    assert registration_key(resolved) == str(LEAKAGE_REMOVER)
    # An older manifest wrote it under `key`; a results tree outlives a release.
    assert registration_key({"key": "name=x--tags=[]--namespace=y"}) == (
        "name=x--tags=[]--namespace=y"
    )
