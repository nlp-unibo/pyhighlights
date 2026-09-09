"""What a corpus looks like before anything trains on it.

Two numbers decide a run's hyperparameters and neither is in the frame: how
long the documents are, which sets a token budget, and how much of a document
its annotation marks, which is what a sparsity target aims at. Both are read
off the **training** split -- an evaluation split's rates are describable
after the fact, never a target, and the model has not seen them.
"""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd

__all__ = ["COLUMNS", "describe"]

#: Column order of the frame :func:`describe` returns.
COLUMNS = [
    "split",
    "rows",
    "annotated",
    "tokens_min",
    "tokens_mean",
    "tokens_median",
    "tokens_p95",
    "tokens_max",
    "highlight_rate",
    "highlights_mean",
]


def _spread(values: np.ndarray) -> Dict[str, float]:
    if not len(values):
        return dict.fromkeys(("min", "mean", "median", "p95", "max"), float("nan"))
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
    }


def describe(splits: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Report document length and annotation density for each split.

    ``highlight_rate`` is marked tokens over all tokens of the annotated rows
    -- the same ratio
    :class:`~pyhighlights.utility.losses.SparsityPenalty` compares its
    ``threshold`` against, so a training split's rate is where that threshold
    comes from. ``highlights_mean`` is marked tokens per annotated row, which
    a rate alone hides: the same rate covers three tokens of thirty and thirty
    of three hundred.

    A split with no annotated row reports ``NaN`` rather than zero. Zero would
    read as an annotation that marks nothing.
    """
    rows = []
    for name, frame in splits.items():
        lengths = frame["tokens"].map(len).to_numpy(dtype=float)
        annotated = frame["highlights"].map(lambda value: value is not None)
        marked = frame.loc[annotated, "highlights"].map(sum).to_numpy(dtype=float)
        covered = lengths[annotated.to_numpy(dtype=bool)]
        rows.append(
            {
                "split": name,
                "rows": len(frame),
                "annotated": int(annotated.sum()),
                **{f"tokens_{key}": value for key, value in _spread(lengths).items()},
                "highlight_rate": (
                    float(marked.sum() / covered.sum())
                    if covered.sum()
                    else float("nan")
                ),
                "highlights_mean": (
                    float(marked.mean()) if len(marked) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)
