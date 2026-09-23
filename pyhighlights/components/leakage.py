"""Leakage analysis: what the splits of a corpus share with each other.

Published splits overlap more often than their papers admit -- every
annotated row of an R2A Hotel aspect also sits in that aspect's training
file -- and nothing downstream can detect it: a model reports highlight
scores on rows it was trained on and the numbers look ordinary. Detection
lives here; repair is a preprocessing step, in
:mod:`pyhighlights.components.preprocessors`.
"""

from __future__ import annotations

import itertools
import re
from typing import Dict, Mapping

import pandas as pd

__all__ = ["LeakageDetector", "duplicates", "leakage", "normalize"]


def normalize(value: object) -> str:
    """One comparable key: collapsed whitespace, stripped, lower-cased.

    ``object`` rather than ``str`` because the column holds whatever the
    loader parsed. A missing value normalizes to the empty string, and an
    empty key is never counted as shared: two rows a corpus left without text
    are two unusable rows rather than a duplicate. They are dropped by
    :func:`~pyhighlights.components.preprocessors.remove_leakage`, which is
    where a row leaves a corpus.
    """
    if value is None or (isinstance(value, float) and value != value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def leakage(
    splits: Mapping[str, pd.DataFrame],
    key: str = "text",
    normalize_keys: bool = True,
) -> pd.DataFrame:
    """Report, for each ordered split pair, how much of the right split the
    left one already contains.

    ``ratio`` is the share of *right* rows found in *left*, so the test row of
    a train/test pair answers "how much of my evaluation set did I train on".
    Keys are whitespace- and case-normalized unless told otherwise, since raw
    equality understates real overlap. An empty key matches nothing, and
    ``size`` counts it all the same: a row with no text is not shared with
    anything, and hiding it would report a share of a corpus that is smaller
    than the one on disk.
    """
    keys = {
        name: frame[key].map(normalize) if normalize_keys else frame[key]
        for name, frame in splits.items()
    }
    seen = {name: set(values) - {""} for name, values in keys.items()}
    rows = []
    for left, right in itertools.permutations(keys, 2):
        overlap = int(keys[right].isin(seen[left]).sum())
        size = len(keys[right])
        rows.append(
            {
                "left": left,
                "right": right,
                "overlap": overlap,
                "size": size,
                "ratio": overlap / size if size else 0.0,
            }
        )
    return pd.DataFrame(rows, columns=["left", "right", "overlap", "size", "ratio"])


def duplicates(
    splits: Mapping[str, pd.DataFrame],
    key: str = "text",
    normalize_keys: bool = True,
) -> Dict[str, int]:
    """Count repeated rows inside each split.

    Keys are whitespace- and case-normalized as in :func:`leakage`, and
    ``normalize_keys=False`` compares them exactly as the corpus spells them.
    An empty key is not a repeat of another empty one.
    """
    counted = {}
    for name, frame in splits.items():
        values = frame[key].map(normalize) if normalize_keys else frame[key]
        counted[name] = int(values[values != ""].duplicated().sum())
    return counted


class LeakageDetector:
    """Reports what a set of splits shares, and refuses splits that share a row.

    Holds no data of its own: every method takes the splits to analyse, so one
    detector serves whichever loader or preprocessing stage is being checked.
    """

    def __init__(self, key: str = "text", normalize_keys: bool = True):
        self.key = key
        self.normalize_keys = normalize_keys

    def report(self, splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Overlap for every ordered split pair."""
        return leakage(splits, key=self.key, normalize_keys=self.normalize_keys)

    def repeats(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, int]:
        """Repeated rows within each split, which :meth:`check` does not read."""
        return duplicates(splits, key=self.key, normalize_keys=self.normalize_keys)

    def check(self, splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Return the report, raising when any split pair shares a row.

        Meant for a test or the top of a run. A corpus that shares rows
        between train and test reports highlight scores on examples the model
        was trained on, and every number downstream is quietly wrong.

        There is no tolerance to set. A shared row is leakage at any rate, and
        a corpus distributed with one -- R2A is -- is read with
        :meth:`report`, which says how much it shares without refusing it.

        **Between splits only.** A split that holds the same row twice passes
        here: :meth:`repeats` is what counts those, and
        :class:`~pyhighlights.components.preprocessors.LeakageRemover` is what
        drops them.
        """
        report = self.report(splits)
        offending = report[report["ratio"] > 0]
        if not offending.empty:
            raise ValueError(f"splits share rows:\n{offending.to_string(index=False)}")
        return report
