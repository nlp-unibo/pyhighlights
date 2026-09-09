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
from typing import Any, Dict, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.preprocessors import Preprocessor

#: The predictions one seed left behind. A glob rather than a name: a run
#: stores one file per seed, and every analyzer here reads all of them.
PREDICTIONS = "predictions-seed=*.pkl"

__all__ = [
    "Analyzer",
    "HighlightPositionAnalyzer",
    "LabelStudioExporter",
    "MetricsAnalyzer",
    "PREDICTIONS",
    "PredictionAnalyzer",
    "escape",
    "label_studio",
    "latex_table",
    "offsets",
    "run_of",
    "reported_head",
    "seed_of",
]


def seed_of(path: Path) -> str:
    """The seed in ``predictions-seed=42.pkl``, or ``"?"`` if it says none."""
    _, _, seed = path.stem.partition("seed=")
    return seed or "?"


def run_of(path: Path, directory: Path) -> str:
    """Which run a predictions file belongs to, as a path under ``directory``.

    The stamp alone does not name a run: a benchmark writes
    ``<name>/<started>`` per task, and two tasks that started in the same
    second would report the same run while being different runs.
    """
    return str(path.parent.relative_to(directory))


def reported_head(masks: np.ndarray) -> np.ndarray:
    """The head a model with several selectors is scored on.

    The reported metrics score the head the aggregator keeps, so every analysis
    reads that one rather than a mixture of heads nothing else reports on.
    """
    return masks[:, 0] if masks.ndim == 3 else masks


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

    A task keeps every run it has ever done, one timestamped directory each, so
    ``latest`` decides which of them the table is about: the most recent run of
    each task by default, every run when asked. Reporting all of them by
    default would grow the table each time a configuration is re-run, and the
    numbers a paper quotes are the last ones measured.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        metrics: Sequence[str] = (),
        split: str = "test",
        pairs: bool = False,
        latest: bool = True,
    ):
        super().__init__(directory)
        self.metrics = list(metrics)
        self.split = split
        self.pairs = pairs
        self.latest = latest

    def reports(self) -> List[Dict[str, Any]]:
        """Every run found, newest last, one per task when ``latest``.

        Grouped by the name the run reported rather than by its directory: the
        stamp is a path component, and a task that has been renamed or moved is
        still the task its results say it is.
        """
        found = {}
        for path in sorted(self.directory.rglob("results.json")):
            report = {"run": path.parent.name, **json.loads(path.read_text())}
            found.setdefault(report.get("name", "?"), []).append(report)
        return [
            report
            for reports in found.values()
            for report in (reports[-1:] if self.latest else reports)
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
                "run": report["run"],
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
    different lengths are comparable. ``absolute`` reports word positions
    instead, which is the other question: a model keying on the first three
    words of every document does that regardless of how long the document is,
    and a share hides it in the first bin of a short document and the first
    tenth of a long one. Columns then cover the first ``bins`` words, and a
    selection past them is counted in the total without a column of its own,
    so the reported shares sum to less than one by however much the tail
    holds.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        pattern: str = PREDICTIONS,
        bins: int = 10,
        absolute: bool = False,
    ):
        super().__init__(directory)
        if bins < 1:
            raise ValueError("bins must be positive")
        self.pattern = pattern
        self.bins = bins
        self.absolute = absolute

    def analyze(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.directory.rglob(self.pattern)):
            positions: Counter = Counter()
            selected = kept = tokens = 0
            for batch in pd.read_pickle(path):
                masks = reported_head(np.asarray(batch["highlight_mask"]))
                valid = np.asarray(batch["mask"])
                for highlights, length_mask in zip(masks, valid):
                    length = int(length_mask.sum())
                    if not length:
                        continue
                    marked = np.flatnonzero(highlights[:length])
                    positions.update(
                        int(index) if self.absolute else int(index / length * self.bins)
                        for index in marked
                    )
                    selected += len(marked)
                    kept += 1
                    tokens += length

            row: Dict[str, Any] = {
                "run": run_of(path, self.directory),
                "seed": seed_of(path),
                "samples": kept,
                "selection_rate": selected / tokens if tokens else 0.0,
            }
            total = sum(positions.values())
            column = "position" if self.absolute else "bin"
            for index in range(self.bins):
                row[f"{column}_{index}"] = positions[index] / total if total else 0.0
            rows.append(row)

        return pd.DataFrame(rows)


class PredictionAnalyzer(Analyzer):
    """What the model selected, in words, one row per sample.

    A stored prediction is token ids and masks: enough to score, unreadable on
    its own. This joins it back to the corpus it came from, so a row says which
    words the selector kept and what the predictor made of them.

    The corpus is not stored beside the predictions -- it would be stored once
    per run and per seed -- so it is reloaded. The run's ``manifest.json`` names
    the loader and the preprocessor that produced it, and those keys are what
    get built here: a corpus loaded from anywhere else is a different corpus.

    Selections are folded from token positions back to words through the
    ``word_ids`` the batch carries, so a subword model reports words like every
    other. A word counts as selected when any of its subtokens was.

    A task builds its loader and its preprocessor from their keys alone, with
    no overrides, so rebuilding those keys rebuilds exactly the corpus the run
    trained against -- an override changes *which* key a task holds, and that
    key is the one the manifest wrote down.

    The registry has to be built before this runs, since it resolves the keys
    the manifest names. Inside a cinnamon script it already is.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        pattern: str = PREDICTIONS,
        split: str = "test",
    ):
        super().__init__(directory)
        self.pattern = pattern
        self.split = split

    def corpus(self, run: Path) -> Dict[int, pd.Series]:
        """The split these predictions were made on, keyed by sample id."""
        settings = json.loads((run / "manifest.json").read_text())["settings"]
        splits = Registry.from_key(
            RegistrationKey.parse(registration_key=settings["loader"]["key"]),
            expected_type=HighlightLoader,
        ).load()
        preprocessor = settings.get("preprocessor")
        if preprocessor is not None:
            splits = Registry.from_key(
                RegistrationKey.parse(registration_key=preprocessor["key"]),
                expected_type=Preprocessor,
            ).process(splits)
        frame = splits[self.split]
        return {int(row.sample_id): row for row in frame.itertuples(index=False)}

    def frames(self) -> Iterator[Tuple[Path, pd.DataFrame]]:
        """One frame per predictions file, with the file that produced it.

        Per file rather than one frame for everything, so a caller that writes
        something back can write it beside the predictions it came from. Every
        seed of a run has its own file and its own predictions of the same
        samples, which is a separate thing to read rather than N copies of one.
        """
        # One corpus per run rather than per seed: every seed of a run was
        # trained on the same split, and loading it again per file is the whole
        # cost of the analysis repeated.
        corpora: Dict[Path, Dict[int, Any]] = {}
        for path in sorted(self.directory.rglob(self.pattern)):
            rows: List[Dict[str, Any]] = []
            run = path.parent
            if run not in corpora:
                corpora[run] = self.corpus(run)
            examples = corpora[run]
            for batch in pd.read_pickle(path):
                masks = reported_head(np.asarray(batch["highlight_mask"]))
                word_ids = np.asarray(batch["word_ids"])
                valid = np.asarray(batch["mask"]) > 0
                predicted = np.asarray(batch["class_logits"])
                if predicted.ndim == 3:
                    predicted = reported_head(predicted)
                predicted = predicted.argmax(-1)

                for index, sample_id in enumerate(batch["sample_ids"]):
                    example = examples.get(int(sample_id))
                    if example is None:
                        # A corpus that no longer holds the sample is a corpus
                        # that changed under the run. Reporting the rest of the
                        # split is more use than refusing all of it.
                        continue
                    tokens = list(example.tokens)
                    words = word_ids[index][valid[index] & (masks[index] > 0)]
                    if words.size and words.max() >= len(tokens):
                        # The corpus places this sample's words differently
                        # than the run did. Folding the selection against it
                        # anyway would print a rationale nothing selected.
                        continue
                    # A padded position belongs to no word and says ``-1``.
                    selected = sorted({int(word) for word in words if word >= 0})
                    rows.append(
                        {
                            "run": run_of(path, self.directory),
                            "seed": seed_of(path),
                            "sample_id": int(sample_id),
                            "label": int(example.label),
                            "predicted": int(predicted[index]),
                            "tokens": tokens,
                            "selected": selected,
                            "rationale": " ".join(tokens[word] for word in selected),
                            "highlights": None
                            if example.highlights is None
                            else [
                                word
                                for word, marked in enumerate(example.highlights)
                                if marked
                            ],
                        }
                    )
            yield path, pd.DataFrame(rows)

    def analyze(self) -> pd.DataFrame:
        frames = [frame for _, frame in self.frames() if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def offsets(tokens: Sequence[str]) -> List[Tuple[int, int]]:
    """Character span of each token in ``" ".join(tokens)``.

    Label Studio addresses a span by character offset into the text it shows,
    and a corpus arrives as words. One space between them is the same
    assumption the text itself is built on, so the two agree by construction.
    """
    spans = []
    start = 0
    for token in tokens:
        spans.append((start, start + len(token)))
        start += len(token) + 1
    return spans


def label_studio(
    frame: pd.DataFrame,
    model_version: str = "pyhighlights",
    labels: Sequence[str] = ("highlight",),
    score: float = 1.0,
) -> List[Dict[str, Any]]:
    """Predicted highlights as Label Studio pre-annotations.

    Reads the columns :class:`PredictionAnalyzer` reports -- ``tokens``,
    ``selected``, ``label``, ``predicted`` -- so it converts any frame carrying
    them, whatever produced it.

    Each selected word becomes one span. The whole document goes in ``data``
    alongside the gold and predicted label, so a reader sees what the model was
    given and what it made of it, not only what it highlighted.
    """
    tasks = []
    for row in frame.itertuples(index=False):
        tokens = list(row.tokens)
        spans = offsets(tokens)
        tasks.append(
            {
                "data": {
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "sample_id": int(row.sample_id),
                    "label": int(row.label),
                    "predicted": int(row.predicted),
                },
                "predictions": [
                    {
                        "model_version": model_version,
                        "score": score,
                        "result": [
                            {
                                "id": f"{row.sample_id}_{word}",
                                "type": "labels",
                                "from_name": "label",
                                "to_name": "text",
                                "value": {
                                    "start": spans[word][0],
                                    "end": spans[word][1],
                                    "score": score,
                                    "text": tokens[word],
                                    "labels": list(labels),
                                },
                            }
                            for word in row.selected
                        ],
                    }
                ],
            }
        )
    return tasks


class LabelStudioExporter(PredictionAnalyzer):
    """Writes each run's predicted highlights where an annotator can read them.

    An expert judging whether a highlight is the right one needs it in front of
    the text, not as a list of word indices. This writes one Label Studio file
    per run, pre-annotated with what the model selected, so the reading is a
    review rather than a fresh annotation.

    ``only`` narrows the export to samples of one gold label, which is what a
    corpus annotated for a rare class needs: the negatives are 97% of it and
    the interesting highlights are all on the positives.

    One file per seed, beside the predictions it came from. A run's seeds are
    separate predictions of the same samples, so merging them would show the
    same sentence once per seed with different words marked, which is not a
    thing to read.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        pattern: str = PREDICTIONS,
        split: str = "test",
        model_version: str = "pyhighlights",
        labels: Sequence[str] = ("highlight",),
        only: int | None = None,
        stem: str = "label-studio",
    ):
        super().__init__(directory=directory, pattern=pattern, split=split)
        self.model_version = model_version
        self.labels = list(labels)
        self.only = only
        self.stem = stem

    def selected(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The rows this export is about."""
        if self.only is None or frame.empty:
            return frame
        return frame[frame["label"] == self.only].reset_index(drop=True)

    def analyze(self) -> pd.DataFrame:
        return self.selected(super().analyze())

    def export(self) -> Dict[Path, pd.DataFrame]:
        """Write one file per predictions file, and say what went in each."""
        written = {}
        for source, frame in self.frames():
            frame = self.selected(frame)
            # Nothing to export is no file rather than an empty one: an empty
            # Label Studio project reads as a project with nothing to review.
            if frame.empty:
                continue
            path = source.parent / f"{self.stem}-seed={seed_of(source)}.json"
            path.write_text(
                json.dumps(
                    label_studio(
                        frame,
                        model_version=self.model_version,
                        labels=self.labels,
                    ),
                    indent=2,
                )
            )
            written[path] = frame
        return written

    def run(self) -> pd.DataFrame:
        # The export already built every frame; analysing again would reload
        # and re-join each corpus for a second time.
        written = self.export()
        for path, frame in written.items():
            print(f"{len(frame)} samples -> {path}")
        return (
            pd.concat(written.values(), ignore_index=True)
            if written
            else pd.DataFrame()
        )
