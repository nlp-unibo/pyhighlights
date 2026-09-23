"""Benchmarks: several tasks, one report.

A paper's table is not one experiment but a grid of them -- every model over
every corpus, each with its own seeds. A benchmark is that grid: a list of task
keys, run in order, and one report collecting what each of them found.

It owns no training logic. Whatever a task does, the benchmark does not need to
know; it runs the key and keeps the result.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.tasks import Task

logger = logging.getLogger(__name__)

__all__ = ["Benchmark"]


class Benchmark:
    """Runs a list of tasks and writes down what each of them reported.

    A task that raises does not take the rest of the grid with it: the failure
    is recorded against that task and the benchmark carries on, since an
    afternoon of runs should not be lost to one bad configuration. ``strict``
    turns that off for a run that must be all-or-nothing.
    """

    def __init__(
        self,
        tasks: Sequence[RegistrationKey[Task]] = (),
        name: str = "benchmark",
        save_path: str | Path | None = None,
        strict: bool = False,
        task_args: Mapping[str, Any] | None = None,
    ):
        if not tasks:
            raise ValueError("a benchmark needs at least one task")
        self.tasks = list(tasks)
        self.name = name
        self.save_path = Path(save_path) if save_path is not None else Path("results")
        self.strict = strict
        # Given to every task the benchmark builds. What it is for is running
        # a grid differently without registering a second one: one batch and
        # one seed to check that every cell holds together, or a smaller batch
        # for a card that cannot fit the registered one. Each task's manifest
        # records what it was built with, so a run overridden this way says so
        # rather than looking like the registered configuration.
        self.task_args = dict(task_args or {})

    @property
    def directory(self) -> Path:
        return self.save_path / self.name

    def build(self, key: RegistrationKey[Task]) -> Task:
        """The task, told to save inside the benchmark's own directory.

        ``save_path`` is the benchmark's own, whatever ``task_args`` says.
        The report is written here and names these tasks, so a task writing
        somewhere else would leave ``benchmark.json`` pointing at results no
        analyzer reading this directory can find.
        """
        return Registry.from_key(
            key,
            expected_type=Task,
            **{**self.task_args, "save_path": str(self.directory)},
        )

    def report(self, results: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """The grid so far, and the settings it was run with.

        ``settings`` is here for the same reason a task writes a manifest: a
        grid run with ``task_args`` produces numbers the registered
        configuration would not, and a report that does not say so cannot be
        told from one that was never overridden.
        """
        return {
            "name": self.name,
            "settings": {
                "tasks": [str(key) for key in self.tasks],
                "strict": self.strict,
                "task_args": self.task_args,
            },
            "tasks": list(results),
            "failed": [item["task"] for item in results if "error" in item],
        }

    def run(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for key in self.tasks:
            task = self.build(key)
            logger.info("%s: running %s", self.name, task.name)
            try:
                # Nested rather than spread: what a task reports is its own,
                # and a task reporting a `task` or a `key` of its own would
                # otherwise rewrite which registration the row claims to be.
                results.append(
                    {"task": task.name, "key": str(key), "result": task.run()}
                )
            except Exception as error:
                if self.strict:
                    raise
                # The grid is worth more than the run that broke: record which
                # one failed and why, and let the rest finish.
                logger.exception("%s: %s failed", self.name, task.name)
                results.append(
                    {
                        "task": task.name,
                        "key": str(key),
                        "error": repr(error),
                        # The log holds the traceback, and a detached run's
                        # log is usually nowhere. The artifact outlives both.
                        "traceback": traceback.format_exc(),
                    }
                )
            # After every task rather than after the grid: a process that is
            # killed mid-run -- a card that falls over, a walltime, the OOM
            # killer -- otherwise leaves a benchmark directory of finished
            # tasks and no report naming any of them.
            self.serialize(self.report(results))

        return self.report(results)

    def serialize(self, report: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "benchmark.json").write_text(json.dumps(report, indent=2))
        return self.directory
