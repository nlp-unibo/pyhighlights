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


def normalize(value: str) -> str:
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
    equality understates real overlap.
    """
    keys = {
        name: frame[key].map(normalize) if normalize_keys else frame[key]
        for name, frame in splits.items()
    }
    rows = []
    for left, right in itertools.permutations(keys, 2):
        seen = set(keys[left])
        overlap = int(keys[right].isin(seen).sum())
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
    """Count repeated rows inside each split."""
    return {
        name: int(
            (frame[key].map(normalize) if normalize_keys else frame[key])
            .duplicated()
            .sum()
        )
        for name, frame in splits.items()
    }


class LeakageDetector:
    """Reports what a set of splits shares, and refuses more than ``tolerance``.

    Holds no data of its own: every method takes the splits to analyse, so one
    detector serves whichever loader or preprocessing stage is being checked.
    """

    def __init__(
        self,
        key: str = "text",
        tolerance: float = 0.0,
        normalize_keys: bool = True,
    ):
        if not 0.0 <= tolerance <= 1.0:
            raise ValueError("tolerance must be between 0 and 1")
        self.key = key
        self.tolerance = tolerance
        self.normalize_keys = normalize_keys

    def report(self, splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
        """Overlap for every ordered split pair."""
        return leakage(splits, key=self.key, normalize_keys=self.normalize_keys)

    def duplicates(self, splits: Mapping[str, pd.DataFrame]) -> Dict[str, int]:
        """Repeated rows within each split."""
        return duplicates(splits, key=self.key, normalize_keys=self.normalize_keys)

    def check(
        self,
        splits: Mapping[str, pd.DataFrame],
        tolerance: float | None = None,
    ) -> pd.DataFrame:
        """Return the report, raising when a split pair exceeds ``tolerance``.

        Meant for a test or the top of a run. A corpus that shares rows
        between train and test reports highlight scores on examples the model
        was trained on, and every number downstream is quietly wrong.
        """
        limit = self.tolerance if tolerance is None else tolerance
        report = self.report(splits)
        offending = report[report["ratio"] > limit]
        if not offending.empty:
            raise ValueError(
                f"splits share rows above the {limit} tolerance:\n"
                f"{offending.to_string(index=False)}"
            )
        return report
