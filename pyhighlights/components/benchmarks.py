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
    ):
        if not tasks:
            raise ValueError("a benchmark needs at least one task")
        self.tasks = list(tasks)
        self.name = name
        self.save_path = Path(save_path) if save_path is not None else Path("results")
        self.strict = strict

    @property
    def directory(self) -> Path:
        return self.save_path / self.name

    def build(self, key: RegistrationKey[Task]) -> Task:
        """The task, told to save inside the benchmark's own directory."""
        return Registry.from_key(key, expected_type=Task, save_path=str(self.directory))

    def run(self) -> Dict[str, Any]:
        results: List[Dict[str, Any]] = []
        for key in self.tasks:
            task = self.build(key)
            logger.info("%s: running %s", self.name, task.name)
            try:
                results.append({"task": task.name, "key": str(key), **task.run()})
            except Exception as error:
                if self.strict:
                    raise
                # The grid is worth more than the run that broke: record which
                # one failed and why, and let the rest finish.
                logger.exception("%s: %s failed", self.name, task.name)
                results.append(
                    {"task": task.name, "key": str(key), "error": repr(error)}
                )

        report = {
            "name": self.name,
            "tasks": results,
            "failed": [item["task"] for item in results if "error" in item],
        }
        self.serialize(report)
        return report

    def serialize(self, report: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "benchmark.json").write_text(json.dumps(report, indent=2))
        return self.directory
