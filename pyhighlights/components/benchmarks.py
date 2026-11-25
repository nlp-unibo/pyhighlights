"""Benchmarks: several tasks, one report.

A paper's table is not one experiment but a grid of them: every model over
every corpus, each with its own seeds. A benchmark is that grid: the task keys a
registry search selects, run in order, and one report collecting what each of
them found. A search follows the variants cinnamon expands, which a fixed list
of keys cannot.

It owns no training logic. Whatever a task does, the benchmark does not need to
know; it runs the key and keeps the result.
"""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.tasks import Task

logger = logging.getLogger(__name__)

__all__ = ["Benchmark"]


class Benchmark:
    """Runs the tasks a registry search selects and records what each reported.

    The search runs in :meth:`run`, after the registry is expanded, so it can
    match variant values. ``task_variants`` matches the value of a variant
    parameter, its default included, and a dotted name such as
    ``model.backbone`` reads into the key a field holds. Keys run sorted by
    their string form, so two runs of one benchmark run in the same order.

    A task that raises does not take the rest of the grid with it: the failure
    is recorded against that task and the benchmark carries on, since an
    afternoon of runs should not be lost to one bad configuration. ``strict``
    turns that off for a run that must be all-or-nothing.
    """

    def __init__(
        self,
        task_name: str = "task",
        task_namespace: str | None = None,
        task_tags: Iterable[str] | None = None,
        task_variants: Mapping[str, Any] | None = None,
        name: str = "benchmark",
        save_path: str | Path | None = None,
        strict: bool = False,
        task_args: Mapping[str, Any] | None = None,
    ):
        self.task_name = task_name
        self.task_namespace = task_namespace
        self.task_tags = frozenset(task_tags) if task_tags is not None else None
        self.task_variants = dict(task_variants) if task_variants else None
        #: The keys the last :meth:`run` resolved; empty before the first one.
        self.tasks: List[RegistrationKey[Task]] = []
        self.name = name
        self.save_path = Path(save_path) if save_path is not None else Path("results")
        self.strict = strict

        self.task_args = dict(task_args or {})

    @property
    def directory(self) -> Path:
        return self.save_path / self.name

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

    def resolve(self) -> List[RegistrationKey[Task]]:
        """The keys the filter matches, sorted by their string form.

        An empty match raises: a typo in a filter would otherwise produce a
        benchmark that runs nothing and reports success.
        """
        keys = Registry.retrieve_keys(
            names=self.task_name,
            namespaces=self.task_namespace,
            tags=self.task_tags,
            variants=self.task_variants,
        )
        if not keys:
            raise ValueError(
                f"{self.name}: no task matches name={self.task_name!r}, "
                f"namespace={self.task_namespace!r}, "
                f"tags={sorted(self.task_tags) if self.task_tags else None!r}, "
                f"variants={self.task_variants!r}"
            )
        return sorted(keys, key=str)

    def run(self) -> Dict[str, Any]:
        self.tasks = self.resolve()
        results: List[Dict[str, Any]] = []
        for key in self.tasks:
            # The benchmark's directory wins over `task_args`: the report names
            # these tasks, and analyzers read their results from this directory.
            task = Registry.from_key(
                key,
                expected_type=Task,
                **{**self.task_args, "save_path": str(self.directory)},
            )
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
            # killed mid-run (a GPU that fails, a walltime, the OOM
            # killer) otherwise leaves a benchmark directory of finished
            # tasks and no report naming any of them.
            self.serialize(self.report(results))

        return self.report(results)

    def serialize(self, report: Mapping[str, Any]) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "benchmark.json").write_text(json.dumps(report, indent=2))
        return self.directory
