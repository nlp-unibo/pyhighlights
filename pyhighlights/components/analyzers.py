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
import logging
from collections import Counter
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np
import pandas as pd
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.components.preprocessors import Preprocessor
from pyhighlights.utility.manifest import registration_key

logger = logging.getLogger(__name__)

#: What :func:`latest_runs` groups: a report, or the directory holding one.
T = TypeVar("T")

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
    "latest_runs",
    "latex_table",
    "offsets",
    "readability",
    "run_of",
    "reported_head",
    "seed_of",
    "separator_of",
    "span_count",
    "task_of",
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


def latest_runs(items: Iterable[Tuple[str, str, T]], latest: bool = True) -> List[T]:
    """One entry per task, newest last, or every entry when ``latest`` is off.

    A task keeps every run it has ever done, one timestamped directory each,
    so a re-run or a requeued job would otherwise read as extra rows: a second
    line in a table, or the same sample exported twice for an annotator.

    Each item arrives as ``(name, stamp, item)``. Grouping is by the name the
    run reported rather than by its directory: the stamp is a path component,
    and a task that has been renamed or moved is still the task its own record
    says it is. Recency is the **stamp** rather than the order the walk
    produced, because those two disagree in exactly that case -- a run under
    ``new-name/2026-09-02`` is walked before one under
    ``old-name/2026-09-01``, and taking the last walked would report the older
    one as current.

    Both analyzers that read a directory of runs group them this way, and they
    differ only in where the name comes from: a metrics report carries it, and
    a predictions file has it in the manifest beside it.
    """
    found: Dict[str, List[Tuple[str, T]]] = {}
    for name, stamp, item in items:
        found.setdefault(name, []).append((stamp, item))
    return [
        item
        for group in found.values()
        for _, item in (
            sorted(group, key=lambda entry: entry[0])[-1:] if latest else group
        )
    ]


def task_of(run: Path) -> str:
    """Which task a run directory belongs to, as its manifest reports it.

    A run that wrote no manifest falls back to the directory the stamp sits
    in, which is where a task writes.
    """
    manifest = run / "manifest.json"
    if not manifest.exists():
        return run.parent.name
    settings = json.loads(manifest.read_text()).get("settings", {})
    return settings.get("name", run.parent.name)


def run_directories(directory: Path, pattern: str, latest: bool = True) -> List[Path]:
    """The run directories holding ``pattern``, newest per task when ``latest``.

    Shared by every analyzer that reads stored predictions, so a re-run is one
    run in each of their reports rather than one in some and two in others.
    """
    candidates = sorted({path.parent for path in directory.rglob(pattern)})
    return sorted(
        latest_runs(((task_of(run), run.name, run) for run in candidates), latest)
    )


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

        The name comes out of ``results.json``, which the run wrote itself.
        :func:`latest_runs` is what does the grouping.
        """
        reports = [
            {"run": path.parent.name, **json.loads(path.read_text())}
            for path in sorted(self.directory.rglob("results.json"))
        ]
        return latest_runs(
            ((report.get("name", "?"), report["run"], report) for report in reports),
            self.latest,
        )

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
                entry = found[name]
                if not isinstance(entry, Mapping) or not {"mean", "std"} <= set(entry):
                    raise ValueError(
                        f"{report['run']} reports {self.split}_{name} as "
                        f"{entry!r}; a summary entry is a mean and a std"
                    )
                mean, std = entry["mean"], entry["std"]
                row[name] = (mean, std) if self.pairs else f"{mean:.4f} +/- {std:.4f}"
            rows.append(row)

        return pd.DataFrame(rows)


def word_axis_map(batch: Mapping[str, Any], masks: np.ndarray) -> np.ndarray | None:
    """``word_ids`` when the selection is over subtokens, ``None`` otherwise.

    The two axes are different widths and a stored batch carries both:
    ``mask`` and ``highlight_true`` are ``[B, W]``, ``word_ids`` and a
    subtoken selection are ``[B, T]``. Which one ``highlight_mask`` lives on
    is the model's ``select_over``, which the predictions do not record, so it
    is read off the width. Equal widths are a vocabulary tokenizer, where the
    map is the identity and either answer gives the same words.
    """
    stored = batch.get("word_ids")
    if stored is None:
        return None
    word_ids = np.asarray(stored)
    return word_ids if masks.shape[1] == word_ids.shape[1] else None


def selected_words(
    highlights: np.ndarray,
    length: int,
    word_ids: np.ndarray | None,
    index: int,
) -> np.ndarray:
    """Which words one sample's selection kept, whichever axis it was made on.

    A subtoken selection is folded through ``word_ids`` and deduplicated: a
    word split into three pieces is one word however many of its pieces were
    selected. Slicing the subtoken mask to the word count instead -- which is
    what this did -- dropped or shifted whatever sat past it, silently and
    without a shape to complain about.

    The word mask is not applied to a folded selection, because it cannot
    narrow one: :class:`~pyhighlights.components.data.HighlightCollator`
    refuses a ``word_id`` outside its own sample's words, so every word this
    returns is a word that sample has, and the padded tail of the word axis is
    unreachable from any subtoken.
    """
    if word_ids is None:
        return np.flatnonzero(highlights[:length])
    kept = (highlights > 0) & (word_ids[index] >= 0)
    return np.unique(word_ids[index][kept])


class HighlightPositionAnalyzer(Analyzer):
    """Where in a document the selector looked, and how much it kept.

    Reads the predictions a task stored. A selector that has learned nothing
    still selects *something*, and position is what tells the two apart: a
    model keying on the first tokens of every document scores like a model that
    found the highlight, until you look at where it selected.

    Positions are word positions, since that is what a selection is made over.

    ``selection_rate`` is the **mean of the per-document rates**, which is what
    :class:`pyhighlights.utility.metrics.SelectionRate` reports and what a
    study's tables are built from. Pooling instead -- all kept words over all
    words -- gives a different number on documents of different lengths, since
    it weights a long document more than a short one, and two quantities under
    one column name is how a table stops being comparable to itself.

    Positions are reported as a share of the document, so documents of
    different lengths are comparable. ``absolute`` reports word positions
    instead, which is the other question: a model keying on the first three
    words of every document does that regardless of how long the document is,
    and a share hides it in the first bin of a short document and the first
    tenth of a long one. Columns then cover the first ``bins`` words, and a
    selection past them is counted in the total without a column of its own,
    so the reported shares sum to less than one by however much the tail
    holds.

    ``latest`` reads the most recent run of each task, as
    :class:`MetricsAnalyzer` and :class:`PredictionAnalyzer` do: a re-run
    task is one block of rows here and one row there, rather than one in some
    reports and two in others.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        pattern: str = PREDICTIONS,
        bins: int = 10,
        absolute: bool = False,
        latest: bool = True,
    ):
        super().__init__(directory)
        if bins < 1:
            raise ValueError("bins must be positive")
        self.pattern = pattern
        self.bins = bins
        self.absolute = absolute
        self.latest = latest

    def analyze(self) -> pd.DataFrame:
        rows = []
        files = [
            path
            for run in run_directories(self.directory, self.pattern, self.latest)
            for path in sorted(run.glob(self.pattern))
        ]
        for path in files:
            positions: Counter = Counter()
            rate = 0.0
            kept = 0
            for batch in pd.read_pickle(path):
                masks = reported_head(np.asarray(batch["highlight_mask"]))
                valid = np.asarray(batch["mask"])
                word_ids = word_axis_map(batch, masks)
                for index, length_mask in enumerate(valid):
                    length = int(length_mask.sum())
                    if not length:
                        continue
                    marked = selected_words(masks[index], length, word_ids, index)
                    positions.update(
                        int(word) if self.absolute else int(word / length * self.bins)
                        for word in marked
                    )
                    rate += len(marked) / length
                    kept += 1

            row: Dict[str, Any] = {
                "run": run_of(path, self.directory),
                "seed": seed_of(path),
                "samples": kept,
                "selection_rate": rate / kept if kept else 0.0,
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

    A selection is made over words, so it is already in the unit a person
    reads: a row lists word positions and the words at them. A run that
    selected over subtokens instead is folded back through the ``word_ids``
    the batch carries, and a word counts as selected when any of its subtokens
    was -- which is why that setting cannot say what the predictor actually
    read, and why it is not the default.

    A task builds its loader and its preprocessor from their keys alone, with
    no overrides, so rebuilding those keys rebuilds exactly the corpus the run
    trained against -- an override changes *which* key a task holds, and that
    key is the one the manifest wrote down.

    The registry has to be built before this runs, since it resolves the keys
    the manifest names. Inside a cinnamon script it already is.

    A sample the corpus no longer holds, or one whose words it places
    differently, is left out rather than refused -- the rest of the split is
    still worth reading. :attr:`skipped` counts both per predictions file, and
    a file that lost anything says so through the module's logger.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        pattern: str = PREDICTIONS,
        split: str = "test",
        latest: bool = True,
    ):
        super().__init__(directory)
        self.pattern = pattern
        self.split = split
        self.latest = latest
        #: Per predictions file, how many rows :meth:`frames` left out and
        #: why. Written on every pass, so it describes the last one.
        self.skipped: Dict[Path, Dict[str, int]] = {}

    def runs(self) -> List[Path]:
        """The run directories to read, newest per task when ``latest``.

        The name comes out of the run's ``manifest.json``, falling back to the
        directory the stamp sits in for a run that wrote none.
        :func:`run_directories` is what does the grouping, for every analyzer
        that reads stored predictions.
        """
        return run_directories(self.directory, self.pattern, self.latest)

    def corpus(self, run: Path) -> Dict[int, pd.Series]:
        """The split these predictions were made on, keyed by sample id."""
        settings = json.loads((run / "manifest.json").read_text())["settings"]
        splits = Registry.from_key(
            RegistrationKey.parse(
                registration_key=registration_key(settings["loader"])
            ),
            expected_type=HighlightLoader,
        ).load()
        preprocessor = settings.get("preprocessor")
        if preprocessor is not None:
            splits = Registry.from_key(
                RegistrationKey.parse(registration_key=registration_key(preprocessor)),
                expected_type=Preprocessor,
            ).process(splits)
        if self.split not in splits:
            raise KeyError(
                f"the run's corpus has no {self.split!r} split; got {sorted(splits)}"
            )
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
        files = [path for run in self.runs() for path in sorted(run.glob(self.pattern))]
        self.skipped = {}
        for path in files:
            rows: List[Dict[str, Any]] = []
            # Both reasons a row is dropped are "the corpus changed under the
            # run", and both used to be silent: a corpus edited enough to miss
            # every sample produced an empty frame and no account of why.
            missing = 0
            misplaced = 0
            run = path.parent
            if run not in corpora:
                corpora[run] = self.corpus(run)
            examples = corpora[run]
            for batch in pd.read_pickle(path):
                masks = reported_head(np.asarray(batch["highlight_mask"]))
                valid = np.asarray(batch["mask"]) > 0
                predicted = np.asarray(batch["class_logits"])
                if predicted.ndim == 3:
                    predicted = reported_head(predicted)
                predicted = predicted.argmax(-1)

                # A selection over words is already word-indexed; one over
                # subtokens is as wide as the encoding and has to be folded.
                word_ids = word_axis_map(batch, masks)

                for index, sample_id in enumerate(batch["sample_ids"]):
                    example = examples.get(int(sample_id))
                    if example is None:
                        # A corpus that no longer holds the sample is a corpus
                        # that changed under the run. Reporting the rest of the
                        # split is more use than refusing all of it.
                        missing += 1
                        continue
                    tokens = list(example.tokens)
                    # `valid` is the word axis, so it may only narrow a
                    # selection made on that axis: combining it with a
                    # subtoken mask compared two different widths, and numpy
                    # refused to broadcast them.
                    words = selected_words(
                        masks[index] * valid[index]
                        if word_ids is None
                        else masks[index],
                        int(valid[index].sum()),
                        word_ids,
                        index,
                    )
                    if words.size and words.max() >= len(tokens):
                        # The corpus places this sample's words differently
                        # than the run did. Folding the selection against it
                        # anyway would print words nothing selected.
                        misplaced += 1
                        continue
                    selected = [int(word) for word in words]
                    rows.append(
                        {
                            "run": run_of(path, self.directory),
                            "seed": seed_of(path),
                            "sample_id": int(sample_id),
                            "label": int(example.label),
                            "predicted": int(predicted[index]),
                            "tokens": tokens,
                            "text": getattr(example, "text", None) or " ".join(tokens),
                            "selected": selected,
                            # Joined the way this corpus joins its tokens: a
                            # character corpus spells `aab`, not `a a b`.
                            "selected_text": separator_of(
                                tokens, getattr(example, "text", None)
                            ).join(tokens[word] for word in selected),
                            "highlights": None
                            if example.highlights is None
                            else [
                                word
                                for word, marked in enumerate(example.highlights)
                                if marked
                            ],
                        }
                    )
            self.skipped[path] = {"missing": missing, "misplaced": misplaced}
            if missing or misplaced:
                logger.warning(
                    "%s: %d samples the corpus no longer holds and %d whose "
                    "words it places differently were left out of %d",
                    run_of(path, self.directory),
                    missing,
                    misplaced,
                    missing + misplaced + len(rows),
                )
            yield path, pd.DataFrame(rows)

    def analyze(self) -> pd.DataFrame:
        frames = [frame for _, frame in self.frames() if not frame.empty]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def span_count(selected: Sequence[int]) -> int:
    """How many contiguous runs a selection is made of."""
    positions = sorted(selected)
    if not positions:
        return 0
    return 1 + sum(
        1 for before, after in zip(positions, positions[1:]) if after - before > 1
    )


def readability(frame: pd.DataFrame, by: str = "label") -> pd.DataFrame:
    """Selection size, rate and span count, per run and per class.

    Over the rows :meth:`PredictionAnalyzer.analyze` returns, which is where a
    selection is already in words and beside the document it came from.

    **Two columns nothing else reports.** A span count, because a rate cannot
    tell two readable phrases from eight scattered fragments -- twenty per cent
    of a clause in two spans is something a person can read, and the same share
    in eight is not. And the split by class, because domain experts asked
    whether the highlights of negative examples differ from those of positive
    ones, and a pooled average over a split that is 97.7% negative reports the
    negative examples' number and calls it the model's.

    ``by`` is ``"label"`` for the annotation and ``"predicted"`` for what the
    model called it. Both are worth reading: the first shows what was missed,
    the second what was invented.
    """
    if frame.empty:
        return pd.DataFrame()
    rows = frame.assign(
        selection_size=frame["selected"].apply(len),
        selection_rate=[
            len(selected) / len(tokens) if len(tokens) else 0.0
            for selected, tokens in zip(frame["selected"], frame["tokens"])
        ],
        spans=frame["selected"].apply(span_count),
    )
    return (
        rows.groupby(["run", by])[["selection_size", "selection_rate", "spans"]]
        .agg(["mean", "count"])
        .reset_index()
    )


def separator_of(tokens: Sequence[str], text: str | None) -> str:
    """What joins this corpus's tokens, read off the text it wrote.

    A corpus of words joins with a space and a corpus of characters joins with
    nothing -- ``ToyLoader`` writes ``"".join(tokens)`` -- so assuming one of
    them spells the other's documents wrongly.
    """
    return "" if text is not None and "".join(tokens) == text else " "


def offsets(tokens: Sequence[str], text: str | None = None) -> List[Tuple[int, int]]:
    """Character span of each token inside ``text``.

    Label Studio addresses a span by character offset into the text it shows,
    so the offsets have to be into the document the corpus wrote rather than
    into a rejoining of its tokens: a character corpus spells ``aab`` where a
    rejoining with spaces spells ``a a b``, and every offset after the first
    would then be wrong.

    ``text`` defaults to ``" ".join(tokens)``, which is what a corpus of words
    holds. Tokens are located in order, so repeated tokens take successive
    occurrences rather than the first one every time.
    """
    text = " ".join(tokens) if text is None else text
    located = []
    cursor = 0
    for token in tokens:
        start = text.find(token, cursor)
        if start < 0:
            raise ValueError(f"token {token!r} does not occur in the text it came from")
        located.append((start, start + len(token)))
        cursor = start + len(token)
    return located


def label_studio(
    frame: pd.DataFrame,
    model_version: str = "pyhighlights",
    labels: Sequence[str] = ("highlight",),
    score: float = 1.0,
) -> List[Dict[str, Any]]:
    """Predicted highlights as Label Studio pre-annotations.

    Reads the columns :class:`PredictionAnalyzer` reports -- ``tokens``,
    ``selected``, ``label``, ``predicted`` and ``text`` -- so it converts any
    frame carrying them, whatever produced it. A frame without ``text`` falls
    back to joining the tokens with a space.

    Each selected word becomes one span. The whole document goes in ``data``
    alongside the gold and predicted label, so a reader sees what the model was
    given and what it made of it, not only what it highlighted.
    """
    tasks = []
    for row in frame.itertuples(index=False):
        tokens = list(row.tokens)
        # The corpus's own text where the frame carries it, since that is what
        # the offsets below address and what an annotator reads.
        text = getattr(row, "text", None) or " ".join(tokens)
        located = offsets(tokens, text)
        tasks.append(
            {
                "data": {
                    "text": text,
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
                                    "start": located[word][0],
                                    "end": located[word][1],
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

    ``only`` narrows the export to samples of one class, which is what a corpus
    annotated for a rare one needs: the negatives are 97% of it and the
    interesting highlights are all on the positives.

    ``column`` decides *which* class that is, and the two answers are different
    questions. ``"label"`` selects the samples that carry the class, and asks
    whether the model found the right words in them. ``"predicted"`` selects
    the samples the model *called* that class, and asks whether the words it
    kept justify the call -- which is the only one of the two available on a
    corpus with no annotation to select by, and the one that surfaces a
    confident mistake. Any column
    :class:`PredictionAnalyzer` reports may be named.

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
        latest: bool = True,
        model_version: str = "pyhighlights",
        labels: Sequence[str] = ("highlight",),
        only: int | None = None,
        column: str = "label",
        stem: str = "label-studio",
    ):
        super().__init__(
            directory=directory, pattern=pattern, split=split, latest=latest
        )
        self.model_version = model_version
        self.labels = list(labels)
        self.only = only
        self.column = column
        self.stem = stem

    def selected(self, frame: pd.DataFrame) -> pd.DataFrame:
        """The rows this export is about."""
        if self.only is None or frame.empty:
            return frame
        if self.column not in frame.columns:
            raise KeyError(
                f"cannot narrow the export by {self.column!r}: a prediction row "
                f"carries {sorted(frame.columns)}"
            )
        return frame[frame[self.column] == self.only].reset_index(drop=True)

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
