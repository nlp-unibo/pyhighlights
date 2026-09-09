"""Preprocessing: everything done to a corpus after it is parsed.

A loader hands back a corpus as distributed. What happens next -- repairing
leaking splits, turning per-annotator judgements into one label and one
highlight vector -- is an editorial choice, and two studies over the same
corpus routinely make it differently. So it is a component: pick one, or
compose several with :class:`Pipeline`, and the choice is named in a
configuration rather than buried in the loader that fetched the files.
"""

from __future__ import annotations

import abc
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.data import COLUMNS
from pyhighlights.components.leakage import normalize

#: Priority runs from the split that must stay intact to the one that can
#: afford to lose rows. The annotated split comes first: it is the only one
#: carrying highlights, so it is the one worth protecting.
PRIORITY = ("test", "val", "train")

RATIONALES = ("majority", "union", "intersection")
TIES = ("drop", "keep")

__all__ = [
    "AnnotationAggregator",
    "ClassWeights",
    "LabelMapper",
    "LeakageRemover",
    "LengthFilter",
    "PRIORITY",
    "Pipeline",
    "Preprocessor",
    "class_weights",
    "remove_leakage",
]


class Preprocessor(abc.ABC):
    """Turns one set of splits into another.

    Implementations take the frames a loader produced and return frames of the
    same shape, so any of them can be chained with any other.
    """

    @abc.abstractmethod
    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Return the processed splits, leaving the input untouched."""

    def __call__(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        return self.process(splits)


def class_weights(labels: Sequence[Any], classes: int | None = None) -> List[float]:
    """Inverse-frequency weights, ``n / (classes * count)`` per class.

    What a weighted cross entropy needs when the classes are not the same size:
    the rarer a class, the more a mistake on it costs, so a model cannot score
    well by never predicting it.

    ``classes`` is the number of classes the model has, and defaults to the
    largest label seen plus one. Pass it wherever the split might not contain
    every class -- an inferred count would silently give the model one output
    fewer than the task has.
    """
    values = [int(label) for label in labels]
    if not values:
        raise ValueError("class weights need at least one label")
    if min(values) < 0:
        raise ValueError("labels must be non-negative class indices")

    classes = max(values) + 1 if classes is None else classes
    counts = Counter(values)
    missing = [index for index in range(classes) if not counts[index]]
    if missing:
        # Not a weight of zero and not one of infinity: a class the split does
        # not contain has no frequency to invert, and either answer would train
        # something the corpus never showed the model.
        raise ValueError(f"classes {missing} have no examples in this split")
    return [len(values) / (classes * counts[index]) for index in range(classes)]


def remove_leakage(
    splits: Mapping[str, pd.DataFrame],
    priority: Sequence[str] = PRIORITY,
    key: str = "text",
    normalize_keys: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Return splits sharing no row, walking them in ``priority`` order.

    Each split keeps only rows no earlier split claimed and no earlier row of
    its own repeated, so the result has neither cross-split leakage nor
    internal duplicates. Splits missing from ``priority`` are handled last, in
    their original order, and the returned mapping keeps the input order.

    ``sample_id`` is renumbered, since it indexes rows within a split.
    """
    order = [name for name in priority if name in splits]
    order += [name for name in splits if name not in order]

    seen: set[str] = set()
    kept = {}
    for name in order:
        frame = splits[name]
        keys = frame[key].map(normalize) if normalize_keys else frame[key]
        keep = ~keys.isin(seen) & ~keys.duplicated()
        seen.update(keys[keep])
        rows = frame[keep].reset_index(drop=True)
        kept[name] = rows.assign(sample_id=range(len(rows)))
    return {name: kept[name] for name in splits}


class LeakageRemover(Preprocessor):
    """Drops the rows one split shares with another, and its own repeats.

    Which split gives a row up is what ``priority`` decides, and it is a
    judgement about the study rather than about the corpus: keeping the
    annotated split whole is right when highlight scores are the result, and
    wrong when the training set is what must be reproduced. Hence a
    preprocessor -- the loader has no business making that call, and splits
    the user built themselves are just as valid an input as the distributed
    ones.

    :attr:`removed` records how many rows each split lost.
    """

    def __init__(
        self,
        priority: Sequence[str] = PRIORITY,
        key: str = "text",
        normalize_keys: bool = True,
    ):
        self.priority = tuple(priority)
        self.key = key
        self.normalize_keys = normalize_keys
        self.removed: Dict[str, int] = {}

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        repaired = remove_leakage(
            splits,
            priority=self.priority,
            key=self.key,
            normalize_keys=self.normalize_keys,
        )
        self.removed = {
            name: len(splits[name]) - len(repaired[name]) for name in splits
        }
        return repaired


class AnnotationAggregator(Preprocessor):
    """Collapses per-annotator labels and rationales into one of each.

    A corpus annotated by several people -- HateXplain has three per post --
    is loaded with every judgement kept, because reducing them is a choice the
    corpus does not make for you:

    - ``label``: majority vote. A post the annotators split three ways has no
      majority; ``ties="drop"`` removes it, as the HateXplain paper does, and
      ``ties="keep"`` resolves it by annotator order.
    - ``highlights``: ``"majority"`` marks a token more than half the
      rationale vectors marked, ``"union"`` any, ``"intersection"`` all.

    Rows whose rationale vectors are all the wrong width, and classes carrying
    no rationale by design, come back all-zero: "no token was marked", not
    "not annotated".
    """

    def __init__(
        self,
        labels: Sequence[str] = (),
        rationale: str = "majority",
        ties: str = "drop",
        annotator_labels: str = "annotator_labels",
        annotator_highlights: str = "annotator_highlights",
    ):
        if rationale not in RATIONALES:
            raise ValueError(f"rationale must be one of {RATIONALES}")
        if ties not in TIES:
            raise ValueError(f"ties must be one of {TIES}")
        self.labels = {name: index for index, name in enumerate(labels)}
        self.rationale = rationale
        self.ties = ties
        self.annotator_labels = annotator_labels
        self.annotator_highlights = annotator_highlights

    def aggregate(self, vectors: Sequence[Sequence[int]], width: int) -> List[int]:
        valid = [list(vector) for vector in vectors if len(vector) == width]
        if not valid:
            return [0] * width
        counts = [sum(flags) for flags in zip(*valid)]
        if self.rationale == "union":
            return [int(count > 0) for count in counts]
        if self.rationale == "intersection":
            return [int(count == len(valid)) for count in counts]
        return [int(count * 2 > len(valid)) for count in counts]

    def label(self, votes: Sequence[str]) -> int | None:
        (name, count), *_ = Counter(votes).most_common()
        if count == 1 and self.ties == "drop":
            return None
        if self.labels and name not in self.labels:
            raise ValueError(f"unexpected label {name}")
        return self.labels[name] if self.labels else int(name)

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for name, frame in splits.items():
            if self.annotator_labels not in frame.columns:
                processed[name] = frame
                continue

            rows = []
            for row in frame.itertuples():
                label = self.label(getattr(row, self.annotator_labels))
                if label is None:
                    continue
                tokens = list(row.tokens)
                rows.append(
                    {
                        "sample_id": len(rows),
                        "text": row.text,
                        "tokens": tokens,
                        "label": label,
                        "highlights": self.aggregate(
                            getattr(row, self.annotator_highlights), len(tokens)
                        ),
                    }
                )
            processed[name] = pd.DataFrame(rows, columns=list(COLUMNS))
        return processed


class Pipeline(Preprocessor):
    """Runs preprocessors in order, each over what the last returned.

    Steps are registration keys rather than instances, so a pipeline is
    something a configuration states -- aggregate the annotations, then repair
    the leakage the aggregation left behind -- and a second study over the
    same corpus states a different one without touching either step.
    """

    def __init__(self, steps: Sequence[RegistrationKey] = ()):
        self.steps = list(steps)
        self.preprocessors: List[Preprocessor] = [
            Registry.from_key(step, expected_type=Preprocessor) for step in self.steps
        ]

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = dict(splits)
        for preprocessor in self.preprocessors:
            processed = preprocessor.process(processed)
        return processed


class LengthFilter(Preprocessor):
    """Drops rows longer than ``max_length`` tokens.

    Truncating would keep the row and lose the tokens, which for a corpus
    scored on highlights means scoring against an annotation whose tail was
    cut off. A study that caps length to bound its compute drops the row
    instead, and says how many it dropped.
    """

    def __init__(self, max_length: int, column: str = "tokens"):
        if max_length < 1:
            raise ValueError("max_length must be positive")
        self.max_length = max_length
        self.column = column
        self.removed: Dict[str, int] = {}

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for name, frame in splits.items():
            keep = frame[self.column].map(len) <= self.max_length
            self.removed[name] = int((~keep).sum())
            processed[name] = frame[keep].reset_index(drop=True)
        return processed


class LabelMapper(Preprocessor):
    """Rewrites label values through a mapping.

    Collapsing classes is an editorial choice like any other -- the GenSPP
    paper folds HateXplain's ``offensive`` into ``normal`` and trains on two
    classes -- and it has to happen before the votes are counted, not after:
    a post two annotators call ``hatespeech`` and one calls ``offensive`` has
    a majority either way, but one where the votes are ``hatespeech``,
    ``offensive`` and ``normal`` has one only once the last two are the same
    class. So ``column`` may name the per-annotator judgements as readily as a
    resolved label, and a list-valued column is mapped element by element.

    A value the mapping does not name is left as it is.
    """

    def __init__(self, mapping: Mapping[Any, Any], column: str = "label"):
        if not mapping:
            raise ValueError("a label mapping needs at least one entry")
        self.mapping = dict(mapping)
        self.column = column

    def convert(self, value):
        if isinstance(value, (list, tuple, pd.Series, np.ndarray)):
            return [self.mapping.get(item, item) for item in value]
        return self.mapping.get(value, value)

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        processed = {}
        for name, frame in splits.items():
            if self.column not in frame.columns:
                raise KeyError(f"{name} has no column {self.column}")
            frame = frame.copy()
            frame[self.column] = frame[self.column].map(self.convert)
            processed[name] = frame
        return processed


class ClassWeights(Preprocessor):
    """Computes a split's class weights and hands every row back unchanged.

    A weight is a number about a corpus, so it is read off the corpus rather
    than typed into a configuration by hand and hoped to still be right. Doing
    it here rather than inside a task makes the reading a named step: it is in
    the pipeline, so it is in the manifest, and it runs over the split the
    study actually trains on -- after whatever filtering and aggregation came
    before it, which is what changes the frequencies.

    Nothing is added to the frames. :attr:`weights` and :attr:`counts` hold
    what it found, and :class:`~pyhighlights.components.tasks.ClassWeightsTask`
    is what writes them down.
    """

    def __init__(self, split: str = "train", classes: int | None = None):
        self.split = split
        self.classes = classes
        self.weights: List[float] = []
        self.counts: Dict[int, int] = {}

    def process(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.split not in splits:
            raise KeyError(f"no {self.split!r} split to weight; got {sorted(splits)}")
        labels = [int(label) for label in splits[self.split]["label"]]
        self.weights = class_weights(labels, self.classes)
        self.counts = {index: labels.count(index) for index in range(len(self.weights))}
        return dict(splits)
