"""Analyzers: what to make of a directory full of results.

A run leaves ``results.json`` files and pickled predictions behind. An analyzer
reads them back and answers one question about them -- what the numbers are,
where the selector looked -- and returns a :class:`pandas.DataFrame` rather
than printing, so the same analyzer serves a notebook, a test and a LaTeX
table.

Nothing here is interactive and nothing plots. An analyzer that asks which
folder you meant cannot run unattended, and a figure is a choice about
presentation that belongs to whoever is writing the paper.
"""

from __future__ import annotations

import abc
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "Analyzer",
    "HighlightPositionAnalyzer",
    "MetricsAnalyzer",
    "escape",
    "latex_table",
]


def escape(value: Any) -> str:
    """Make a name safe to typeset. An unescaped ``_`` is a subscript."""
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_")


def latex_table(
    frame: pd.DataFrame, precision: int = 2, percentage: bool = True
) -> str:
    r"""The rows of a table body, ``&``-separated and ``\\``-terminated.

    A ``(mean, std)`` pair -- what ``MetricsAnalyzer(pairs=True)`` reports --
    becomes ``$12.34_{\pm 0.56}$``. Anything else is escaped and written as it
    is: metric names carry underscores, and LaTeX reads those as subscripts.

    The header row is included; the ``tabular`` wrapper is not, since its
    column specification belongs to the table this goes into.
    """
    scale = 100 if percentage else 1

    def cell(value: Any) -> str:
        if isinstance(value, tuple):
            mean, std = value
            return (
                f"${mean * scale:.{precision}f}"
                + r"_{\pm "
                + f"{std * scale:.{precision}f}"
                + "}$"
            )
        return escape(value)

    rows = [" & ".join(escape(column) for column in frame.columns)]
    rows += [
        " & ".join(cell(value) for value in row)
        for row in frame.itertuples(index=False)
    ]
    return " \\\\\n".join(rows) + " \\\\"


class Analyzer(abc.ABC):
    """Reads a results directory and reports on it."""

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory is not None else Path("results")

    @abc.abstractmethod
    def analyze(self) -> pd.DataFrame:
        """The report, as a frame."""

    def run(self) -> pd.DataFrame:
        report = self.analyze()
        print(report.to_string(index=False))
        return report


class MetricsAnalyzer(Analyzer):
    """One row per task, one column per metric, ``mean ± std`` across seeds.

    Walks every ``results.json`` beneath the directory, so it reads one task or
    a whole benchmark without being told which. ``metrics`` selects and orders
    the columns; left empty, every metric found is reported.

    A metric a task did not measure is ``-`` rather than missing, since a grid
    of models rarely reports exactly the same set: an unannotated corpus has no
    highlight F1 to give.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        metrics: Sequence[str] = (),
        split: str = "test",
        pairs: bool = False,
    ):
        super().__init__(directory)
        self.metrics = list(metrics)
        self.split = split
        self.pairs = pairs

    def reports(self) -> List[Dict[str, Any]]:
        return [
            json.loads(path.read_text())
            for path in sorted(self.directory.rglob("results.json"))
        ]

    def analyze(self) -> pd.DataFrame:
        rows = []
        for report in self.reports():
            summary = report.get("summary", {})
            prefix = f"{self.split}_"
            found = {
                name[len(prefix) :]: value
                for name, value in summary.items()
                if name.startswith(prefix)
            }
            names = self.metrics or sorted(found)
            row: Dict[str, Any] = {
                "task": report.get("name", "?"),
                "seeds": len(report.get("seeds", [])),
            }
            for name in names:
                if name not in found:
                    row[name] = "-"
                    continue
                mean, std = found[name]["mean"], found[name]["std"]
                row[name] = (mean, std) if self.pairs else f"{mean:.4f} +/- {std:.4f}"
            rows.append(row)

        return pd.DataFrame(rows)


class HighlightPositionAnalyzer(Analyzer):
    """Where in a document the selector looked, and how much it kept.

    Reads the predictions a task stored. A selector that has learned nothing
    still selects *something*, and position is what tells the two apart: a
    model keying on the first tokens of every document scores like a model that
    found the rationale, until you look at where it selected.

    Positions are reported as a share of the document, so documents of
    different lengths are comparable.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        filename: str = "predictions.pkl",
        bins: int = 10,
    ):
        super().__init__(directory)
        if bins < 1:
            raise ValueError("bins must be positive")
        self.filename = filename
        self.bins = bins

    def analyze(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.directory.rglob(self.filename)):
            positions: Counter = Counter()
            selected = kept = tokens = 0
            for batch in pd.read_pickle(path):
                masks = np.asarray(batch["highlight_mask"])
                valid = np.asarray(batch["mask"])
                if masks.ndim == 3:
                    # A model with several selectors stores one mask per head.
                    # The reported metrics score the head the aggregator keeps,
                    # so the analysis reads that one rather than a mixture of
                    # heads nothing else reports on.
                    masks = masks[:, 0]
                for highlights, length_mask in zip(masks, valid):
                    length = int(length_mask.sum())
                    if not length:
                        continue
                    marked = np.flatnonzero(highlights[:length])
                    positions.update(
                        int(index / length * self.bins) for index in marked
                    )
                    selected += len(marked)
                    kept += 1
                    tokens += length

            row: Dict[str, Any] = {
                "run": path.parent.name,
                "samples": kept,
                "selection_rate": selected / tokens if tokens else 0.0,
            }
            total = sum(positions.values())
            for index in range(self.bins):
                row[f"bin_{index}"] = positions[index] / total if total else 0.0
            rows.append(row)

        return pd.DataFrame(rows)
