"""Shortcut analysis: what predicts the label without reading the evidence.

A corpus is a control only while the evidence it annotates is the only thing
that solves it. If some other feature separates the classes, a model can score
well on the task and badly on the explanation, and nothing downstream can tell
the two apart -- which is the failure a synthetic corpus exists to rule out.

The toy corpus is the case this was written for. Its released quality check
scored three hand-picked string-matching baselines by highlight F1, which asks
whether a selection *matches the annotation* rather than whether it *solves
the task*: a competing n-gram can score near zero against the annotation and
still predict the class perfectly, and that is exactly the shortcut the check
is supposed to exclude. What is asked here is the other question, over every
n-gram the corpus actually contains.

The same scan answers it of a real corpus. Whether punctuation predicts an
unfair ToS clause is this question, asked of words instead of characters.

**What a clean report does and does not say.** No scan proves the annotated
evidence is the *only* solution: any feature fine enough to index the sample
separates it. The claim is bounded, and the bound is the feature family --
single n-grams up to ``max_length``, and sequence length. A conjunction of two
n-grams that neither one predicts alone is outside it, and is what
:func:`ablated` is for: remove the evidence and re-run, and nothing expressible
in *any* feature family should be left to find.

Every score is read against a **permuted control**. With thousands of features
the best one beats the majority baseline on chance alone, so the threshold is
not the baseline but the best score any feature reaches once the labels are
shuffled.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "ShortcutDetector",
    "ablated",
    "incidence",
    "lengths",
    "ngrams",
    "scan",
]


def incidence(
    documents: Iterable[Sequence[str]], max_length: int = 4, separator: str = ""
) -> Dict[str, List[int]]:
    """Which documents hold each n-gram, up to ``max_length`` tokens.

    A document counts once for an n-gram it repeats: the question is whether
    the n-gram is *there*, and a count is a different feature.

    ``separator`` joins the tokens for display -- empty for a character corpus
    like the toy one, a space for a corpus of words.
    """
    if max_length < 1:
        raise ValueError("max_length is at least 1")
    found: Dict[str, List[int]] = {}
    for index, tokens in enumerate(documents):
        tokens = list(tokens)
        seen = {
            separator.join(tokens[start : start + width])
            for width in range(1, max_length + 1)
            for start in range(len(tokens) - width + 1)
        }
        for gram in seen:
            found.setdefault(gram, []).append(index)
    return found


def _accuracy(present: np.ndarray, totals: np.ndarray) -> float:
    """Best single rule over one binary feature: a class present, a class absent."""
    absent = totals - present
    return float(present.max() + absent.max()) / float(totals.sum())


def _mutual_information(present: np.ndarray, totals: np.ndarray) -> float:
    size = totals.sum()
    joint = np.vstack([totals - present, present]).astype(float) / size
    feature = joint.sum(axis=1, keepdims=True)
    label = joint.sum(axis=0, keepdims=True)
    expected = feature * label
    # 0 log 0 is 0, and a feature every document has makes a zero row.
    nonzero = (joint > 0) & (expected > 0)
    return float((joint[nonzero] * np.log(joint[nonzero] / expected[nonzero])).sum())


def scan(
    features: Mapping[str, Sequence[int]],
    labels: Sequence[int],
    seed: int = 0,
    permutations: int = 30,
) -> pd.DataFrame:
    """Score every feature by the best rule over it, against a permuted control.

    ``features`` maps a name to the document indices that hold it. Returns one
    row per feature -- ``support``, ``accuracy``, ``permuted``,
    ``mutual_information`` -- sorted by accuracy, with ``baseline`` (the
    majority class) carried on every row so a number is readable on its own.
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        raise ValueError("nothing to scan")
    classes = int(labels.max()) + 1
    totals = np.bincount(labels, minlength=classes)
    generator = random.Random(seed)
    controls = []
    for _ in range(permutations):
        shuffled = labels.copy()
        generator.shuffle(shuffled)
        controls.append(shuffled)

    rows = []
    for name, holders in features.items():
        holders = np.asarray(holders, dtype=int)
        present = np.bincount(labels[holders], minlength=classes)
        rows.append(
            {
                "feature": name,
                "support": int(holders.size),
                "accuracy": _accuracy(present, totals),
                "permuted": max(
                    _accuracy(np.bincount(control[holders], minlength=classes), totals)
                    for control in controls
                ),
                "mutual_information": _mutual_information(present, totals),
            }
        )
    report = pd.DataFrame(
        rows,
        columns=[
            "feature",
            "support",
            "accuracy",
            "permuted",
            "mutual_information",
        ],
    )
    report["baseline"] = float(totals.max()) / float(totals.sum())
    return report.sort_values("accuracy", ascending=False, ignore_index=True)


def ngrams(
    frame: pd.DataFrame,
    max_length: int = 4,
    separator: str = "",
    tokens: str = "tokens",
    label: str = "label",
    seed: int = 0,
    permutations: int = 30,
) -> pd.DataFrame:
    """Every n-gram in the corpus, scored by how well its presence predicts."""
    return scan(
        incidence(frame[tokens], max_length=max_length, separator=separator),
        frame[label].to_numpy(),
        seed=seed,
        permutations=permutations,
    )


def lengths(
    frame: pd.DataFrame,
    tokens: str = "tokens",
    label: str = "label",
    seed: int = 0,
    permutations: int = 30,
) -> pd.DataFrame:
    """How well ``len(tokens) >= t`` predicts, over every threshold there is.

    The channel an n-gram scan cannot see. Patterns of different lengths in a
    corpus of variable-length documents make the document's own length say
    which class it is, and no feature over its content is involved.
    """
    sizes = frame[tokens].map(len).to_numpy()
    thresholds = np.unique(sizes)[1:]
    return scan(
        {f"length >= {int(t)}": np.flatnonzero(sizes >= t) for t in thresholds},
        frame[label].to_numpy(),
        seed=seed,
        permutations=permutations,
    )


def ablated(
    frame: pd.DataFrame, filler: str = "\u25ae", separator: str = ""
) -> pd.DataFrame:
    """The corpus with every annotated position replaced by one filler token.

    Removing the evidence is the decisive test: whatever is left cannot be the
    thing the corpus is about, so a scan that finds anything here has found a
    shortcut -- and unlike a scan of the corpus itself, this one is not bounded
    by a feature family, because there is nothing left for any family to find.

    The replacement is a token the alphabet does not contain, so the hole
    cannot spell anything; its *width* is preserved, which keeps the document's
    length out of the comparison.

    ``separator`` rejoins ``text`` the way the corpus spells it -- empty for a
    character corpus, a space for words. The scan reads ``tokens`` and never
    ``text``, so this only decides whether the returned frame is legible, but
    an ablated word corpus that reads ``the\u25aebrownfox`` is a frame nobody can
    check by eye.
    """
    out = frame.copy()
    out["tokens"] = [
        [filler if flag else token for token, flag in zip(row.tokens, row.highlights)]
        for row in frame.itertuples()
    ]
    out["text"] = out["tokens"].map(separator.join)
    return out


class ShortcutDetector:
    """Reports what predicts a corpus's label, and refuses an unexplained winner.

    Holds no data of its own, like
    :class:`~pyhighlights.components.leakage.LeakageDetector`: every method
    takes the frame to analyse.
    """

    def __init__(
        self,
        max_length: int = 4,
        separator: str = "",
        tokens: str = "tokens",
        label: str = "label",
        seed: int = 0,
        permutations: int = 30,
    ):
        self.max_length = max_length
        self.separator = separator
        self.tokens = tokens
        self.label = label
        self.seed = seed
        #: Label shuffles the threshold is the best of. The threshold is the
        #: largest score any feature reaches on any shuffle, which makes
        #: :meth:`check` a permutation test on the maximum at a level of about
        #: ``1 / permutations`` -- at 10 a clean corpus failed roughly a tenth
        #: of the time. The n-grams are counted once whatever this is, so more
        #: shuffles cost little.
        self.permutations = permutations

    def report(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Every n-gram and every length threshold, in one ranking."""
        options = {
            "tokens": self.tokens,
            "label": self.label,
            "seed": self.seed,
            "permutations": self.permutations,
        }
        return pd.concat(
            [
                ngrams(
                    frame,
                    max_length=self.max_length,
                    separator=self.separator,
                    **options,
                ),
                lengths(frame, **options),
            ],
            ignore_index=True,
        ).sort_values("accuracy", ascending=False, ignore_index=True)

    def check(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Ablate the annotated evidence, then refuse anything that still predicts.

        **The gate is the ablation, not the ranking.** A scan of the corpus
        itself cannot be a gate: the annotated patterns are meant to predict,
        and so is anything that co-occurs with them, so the ranking is full of
        features that are supposed to be there. Worse, a pattern shared by two
        of three classes still separates the third by its *absence* -- being
        shared makes a pattern insufficient, not uninformative.

        What the corpus has to guarantee is the other direction: with the
        evidence gone, nothing is left. That claim is not bounded by a feature
        family, because there is nothing for any family to find.

        The threshold is the best score any feature reaches on shuffled labels,
        which is the multiple-comparison control: with thousands of features the
        best of them beats the majority baseline by chance, and a real shortcut
        is one that beats what chance already offers.

        That makes this a permutation test on the maximum, at a level of about
        ``1 / permutations``, so **a small corpus fails it occasionally without
        anything being wrong**. The message carries the margin for that reason:
        a real shortcut clears the threshold by a distance and holds as the
        corpus grows, where noise clears it by a hair and decays towards the
        baseline. Re-run on more rows before believing a narrow failure.
        """
        report = self.report(ablated(frame, separator=self.separator))
        threshold = report["permuted"].max()
        offending = report[report["accuracy"] > threshold].copy()
        if not offending.empty:
            offending["margin"] = offending["accuracy"] - threshold
            raise ValueError(
                f"{len(offending)} features predict the label with the "
                f"highlight removed, above the permuted best of "
                f"{threshold:.4f} (baseline {report['baseline'].iloc[0]:.4f}, "
                f"widest margin {offending['margin'].max():.4f}):\n"
                f"{offending.head(10).to_string(index=False)}"
            )
        return report
