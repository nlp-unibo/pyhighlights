"""Selection codes: whether a selection holds the evidence or a code for it.

A select-then-predict model can classify well from selections that do not
contain the evidence. Its predictor learned from the selector's own
selections, so it can decode the label from any selection the selector makes
consistently for one class. On the toy corpus, a lone ``b`` for every ``abc``
document is enough. The model's accuracy and its own sufficiency score both
look sound, and neither asks whether the evidence was selected.

This module asks the annotation instead, so it needs a corpus with highlight
annotations. It reads a frame with one row per document and these columns:

``run`` and ``seed``
    Which run and seed the row belongs to.
``tokens``
    The document's tokens.
``selected``
    The positions the model selected.
``highlights``
    The annotated positions, or ``None`` for a document without annotation.
``label`` and ``predicted``
    The gold label and the model's prediction.

:class:`~pyhighlights.components.analyzers.PredictionAnalyzer` writes exactly
these columns, and :class:`SelectionCodeAnalyzer` reads a results tree
through it.

**Complete** means that every annotated position is selected. A corpus can
also ask that no other class's evidence is selected as a contiguous run. That
is ``exclusive``, for a corpus such as the toy one, where each class has one
highlight and every document contains fragments of the others. The check then
rejects a selection that spells two highlights at once.

**Complete in order** keeps the selected tokens in document order and drops
their positions. It asks whether the annotated tokens appear among them as a
contiguous sequence. It is the only view available for a source that stores
the selected tokens without their positions.

**Code** is the most frequent selection of each class, in tokens. Its length
summed over classes compares runs that settled on codes of different size.

This is not the ``sufficiency`` of
:mod:`pyhighlights.components.faithfulness`. Sufficiency asks the model's own
predictor, which can read a code. A complete selection has to contain what the
annotation marks, whatever the predictor makes of it.

The annotation bounds the claim. On a corpus whose annotators disagree, such
as HateXplain, an incomplete selection may still hold evidence that the
annotation left out.

**Mask probes** need no annotation. A logistic regression reads only the shape
of each selection: the document length, the number of positions kept, the
kept share, and a normalized histogram of where the kept positions sit. It is
asked to reproduce the model's own decisions under five-fold cross-validation.
A probe that succeeds far above its permuted control shows that the decision
is recoverable from the mask without its text, which is a code carried by
count or position. The probes need ``pyhighlights[analysis]``.

**Selection entropy** is the Shannon entropy of the tokens a model selects,
following Treviso and Martins (2020), *The Explanation Game: Towards Prediction
Explainability through Sparse Communication*, BlackboxNLP, Appendix F.2. A
code repeats a few tokens per class, so its entropy is low. It is reported
beside the entropy of random selections of the same size in the same
documents, which is its reference.

**Evaluator readings** ask an evaluator that no selector trained. An
:class:`~pyhighlights.components.models.spp.evalx.EvalX` model learns the
label from random selections of the input, so it cannot have learned any
model's code. :func:`evaluator_readings` reports its accuracy on each run's
selections, beside its accuracy on the full text and on the annotation. A
selection the evaluator cannot read while the model's own predictor can is a
code. Unlike the complete-highlight rate, this needs no annotation.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.analyzers import PREDICTIONS, PredictionAnalyzer

__all__ = [
    "SelectionCodeAnalyzer",
    "class_highlights",
    "codes",
    "complete",
    "complete_in_order",
    "contains",
    "evaluator_readings",
    "evaluators",
    "gold_recall",
    "kept_by_label",
    "mask_features",
    "mask_probes",
    "runs",
    "selection_entropy",
    "summarize",
]

#: Bins of the position histogram.
BINS = 10
#: Shorter documents have no position geometry, and the histogram divides by
#: the length minus one.
MIN_TOKENS = 4
#: Feature columns of :func:`mask_features` that each probe channel reads.
CHANNELS = {
    "count": [1],
    "length": [0],
    "count_length": [0, 1, 2],
    "position": list(range(3, 3 + BINS)),
    "mask": list(range(3 + BINS)),
}


def contains(sequence: Sequence, part: Sequence) -> bool:
    """Whether ``part`` occurs in ``sequence`` as a contiguous subsequence."""
    width = len(part)
    if width == 0:
        return False
    return any(
        list(sequence[start : start + width]) == list(part)
        for start in range(len(sequence) - width + 1)
    )


def runs(selected: Iterable[int]) -> List[List[int]]:
    """The selected positions, split into maximal runs of adjacent ones."""
    found: List[List[int]] = []
    for position in sorted(set(selected)):
        if found and position == found[-1][-1] + 1:
            found[-1].append(position)
        else:
            found.append([position])
    return found


def complete(
    tokens: Sequence[str],
    selected: Iterable[int],
    highlights: Sequence[int],
    rivals: Iterable[Sequence[str]] = (),
) -> bool:
    """Every annotated position is selected, and no rival is a selected run.

    A rival is another class's evidence as a token sequence. It counts as
    selected when it occurs inside one run of adjacent selected positions.
    """
    chosen = set(selected)
    if not highlights or any(position not in chosen for position in highlights):
        return False
    pieces = [[tokens[position] for position in run] for run in runs(chosen)]
    return not any(contains(piece, rival) for piece in pieces for rival in rivals)


def complete_in_order(
    tokens: Sequence[str],
    selected: Iterable[int],
    highlights: Sequence[int],
    rivals: Iterable[Sequence[str]] = (),
) -> bool:
    """The annotated tokens occur among the selected ones, and no rival does.

    The selected tokens are kept in document order and their positions are
    dropped, so the annotated tokens may be matched elsewhere in the document.
    """
    kept = [tokens[position] for position in sorted(set(selected))]
    gold = [tokens[position] for position in sorted(highlights)]
    if not contains(kept, gold):
        return False
    return not any(contains(kept, rival) for rival in rivals)


def gold_recall(selected: Iterable[int], highlights: Sequence[int]) -> float:
    """The share of annotated positions that were selected."""
    if not highlights:
        return float("nan")
    chosen = set(selected)
    return sum(position in chosen for position in highlights) / len(highlights)


def class_highlights(frame: pd.DataFrame) -> Dict[int, Tuple[str, ...]]:
    """Each class's single highlight, as a token sequence.

    Raises ``ValueError`` when a class has no annotated document, or more than
    one distinct highlight. ``exclusive`` checks need exactly one per class.
    """
    found: Dict[int, set] = {}
    for row in frame.itertuples():
        if row.highlights:
            sequence = tuple(row.tokens[position] for position in row.highlights)
            found.setdefault(int(row.label), set()).add(sequence)
    labels = set(int(label) for label in frame["label"])
    if set(found) != labels:
        raise ValueError(
            f"classes {sorted(labels - set(found))} have no annotated document"
        )
    several = {
        label: len(sequences)
        for label, sequences in found.items()
        if len(sequences) > 1
    }
    if several:
        raise ValueError(
            f"exclusive needs one highlight per class, and these classes have "
            f"more: {several}"
        )
    return {label: next(iter(sequences)) for label, sequences in found.items()}


def summarize(frame: pd.DataFrame, exclusive: bool = False) -> pd.DataFrame:
    """One row per run and seed.

    ``accuracy`` is the model's own accuracy over every document. The three
    selection measures average over annotated documents only, and
    ``annotated`` counts them. ``code_size`` is the length of the most
    frequent selection, summed over classes.
    """
    rivals = _rivals(frame, exclusive)
    rows = []
    for (run, seed), group in frame.groupby(["run", "seed"], sort=True):
        annotated = group[group["highlights"].map(bool)]
        others = [rivals.get(int(label), ()) for label in annotated["label"]]
        triples = list(
            zip(annotated["tokens"], annotated["selected"], annotated["highlights"])
        )
        rows.append(
            {
                "run": run,
                "seed": seed,
                "samples": len(group),
                "annotated": len(annotated),
                "accuracy": float((group["predicted"] == group["label"]).mean()),
                "complete": _mean(
                    complete(*triple, rival) for triple, rival in zip(triples, others)
                ),
                "complete_in_order": _mean(
                    complete_in_order(*triple, rival)
                    for triple, rival in zip(triples, others)
                ),
                "gold_recall": _mean(
                    gold_recall(selected, highlights)
                    for _, selected, highlights in triples
                ),
                "code_size": int(codes(group)["size"].sum()),
            }
        )
    return pd.DataFrame(rows)


def codes(frame: pd.DataFrame) -> pd.DataFrame:
    """The most frequent selection of each class, per run and seed.

    ``code`` is the selection's tokens, ``size`` its length in tokens, and
    ``share`` the fraction of the class's documents that selected exactly it.
    A tie goes to the selection seen first.
    """
    rows = []
    for (run, seed, label), group in frame.groupby(["run", "seed", "label"], sort=True):
        counts = Counter(
            tuple(tokens[position] for position in sorted(set(selected)))
            for tokens, selected in zip(group["tokens"], group["selected"])
        )
        code, count = counts.most_common(1)[0]
        rows.append(
            {
                "run": run,
                "seed": seed,
                "label": label,
                "code": code,
                "size": len(code),
                "share": count / len(group),
            }
        )
    return pd.DataFrame(rows, columns=["run", "seed", "label", "code", "size", "share"])


def mask_features(tokens: Sequence[str], selected: Iterable[int]) -> List[float]:
    """The shape of one selection, without its text.

    The document length, the number of positions kept, the kept share, and a
    histogram of where the kept positions sit over ``BINS`` bins. The histogram
    is normalized to sum to one, so that the position channel cannot carry the
    count.
    """
    length = len(tokens)
    chosen = np.array(sorted(set(selected)), dtype=float)
    where = chosen / max(1, length - 1)
    histogram, _ = np.histogram(where, bins=BINS, range=(0, 1))
    histogram = histogram / max(1, histogram.sum())
    return [length, len(chosen), len(chosen) / length, *histogram]


def kept_by_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Positions kept and kept share, per run, seed and gold label.

    A gap between labels is the simplest code: the selector signals the label
    through how much it keeps.
    """
    shaped = frame.assign(
        kept=[len(set(selected)) for selected in frame["selected"]],
        share=[
            len(set(selected)) / len(tokens)
            for tokens, selected in zip(frame["tokens"], frame["selected"])
        ],
    )
    return (
        shaped.groupby(["run", "seed", "label"], sort=True)[["kept", "share"]]
        .mean()
        .reset_index()
    )


def mask_probes(frame: pd.DataFrame, random_state: int = 0) -> pd.DataFrame:
    """Whether the model's decisions are recoverable from mask shape alone.

    One row per run and seed. Documents shorter than ``MIN_TOKENS`` are left
    out. Every score is a cross-validated F1 of a class-balanced logistic
    regression over :func:`mask_features`: positive-class F1 for two classes
    and macro F1 otherwise.

    ``f1`` reproduces the decisions from every feature, and ``permuted_f1``
    does the same against the decisions shuffled, which is its control.
    ``mcc`` is the Matthews correlation of the first. ``on_label`` predicts the
    gold label instead: a probe that reproduces the model far better than the
    label shows that the decision follows the mask rather than the document.
    ``channel_<name>`` repeats the probe on one group of features in
    ``CHANNELS``. ``random_state`` seeds the permutation of each run and seed.
    """
    try:
        from sklearn.metrics import f1_score, matthews_corrcoef
    except ImportError as error:
        raise ImportError("mask probes require pyhighlights[analysis]") from error

    rows = []
    for (run, seed), group in frame.groupby(["run", "seed"], sort=True):
        group = group[group["tokens"].map(len) >= MIN_TOKENS]
        features = np.array(
            [
                mask_features(tokens, selected)
                for tokens, selected in zip(group["tokens"], group["selected"])
            ],
            dtype=float,
        )
        decisions = group["predicted"].to_numpy()
        labels = group["label"].to_numpy()
        average = "binary" if len(np.union1d(decisions, labels)) <= 2 else "macro"
        shuffled = np.random.default_rng(random_state).permutation(decisions)
        predicted = _probe(features, decisions)
        row = {
            "run": run,
            "seed": seed,
            "documents": len(group),
            "f1": f1_score(decisions, predicted, average=average),
            "permuted_f1": f1_score(
                shuffled, _probe(features, shuffled), average=average
            ),
            "mcc": matthews_corrcoef(decisions, predicted),
            "on_label": f1_score(labels, _probe(features, labels), average=average),
        }
        for name, columns in CHANNELS.items():
            row[f"channel_{name}"] = f1_score(
                decisions, _probe(features[:, columns], decisions), average=average
            )
        rows.append(row)
    return pd.DataFrame(rows)


def selection_entropy(
    frame: pd.DataFrame, per_class: bool = True, random_state: int = 0
) -> pd.DataFrame:
    """Entropy, in bits, of the selected tokens per run, seed and class.

    ``entropy`` counts every selected token of the group's documents.
    ``random_entropy`` does the same for a random selection of the same size in
    each document, drawn with ``random_state``. ``gold_entropy`` counts the
    annotated tokens of the group's annotated documents, and is ``nan`` without
    annotation. Evidence that repeats few tokens, such as a toy highlight, has
    a low entropy too, so a low ``entropy`` reads as a code only below
    ``gold_entropy``. A high one does not rule a code out, since a size code
    can keep arbitrary tokens. ``kept`` is the number of selected tokens. With
    ``per_class`` false, each run and seed is one group.
    """
    keys = ["run", "seed", "label"] if per_class else ["run", "seed"]
    rng = np.random.default_rng(random_state)
    rows = []
    for key, group in frame.groupby(keys, sort=True):
        chosen: Counter = Counter()
        drawn: Counter = Counter()
        gold: Counter = Counter()
        for tokens, selected, highlights in zip(
            group["tokens"], group["selected"], group["highlights"]
        ):
            positions = sorted(set(selected))
            chosen.update(tokens[position] for position in positions)
            random = rng.choice(len(tokens), size=len(positions), replace=False)
            drawn.update(tokens[position] for position in random)
            gold.update(tokens[position] for position in highlights or ())
        rows.append(
            {
                **dict(zip(keys, key)),
                "kept": sum(chosen.values()),
                "entropy": _entropy(chosen),
                "random_entropy": _entropy(drawn),
                "gold_entropy": _entropy(gold) if gold else float("nan"),
            }
        )
    columns = [*keys, "kept", "entropy", "random_entropy", "gold_entropy"]
    return pd.DataFrame(rows, columns=columns)


def evaluator_readings(frame: pd.DataFrame, evaluator, loader) -> pd.DataFrame:
    """How a trained evaluator classifies each run and seed's selections.

    One row per run and seed. ``documents`` counts the frame's documents that
    ``loader`` holds, matched by sample id. ``on_selection`` is the
    evaluator's accuracy against the gold label from the model's selection
    alone, and ``agrees`` is how often it reproduces the model's own
    prediction from it. ``on_full`` reads every valid position. ``on_highlight``
    reads the annotated positions, over the ``annotated`` documents that have
    an annotation. The two references bound what a selection can be read as.

    Each accuracy has a ``p_`` column beside it: the evaluator's mean
    probability of the gold label. Read it first where the evidence is
    ambiguous. A selection that two classes share has a gold probability
    near one half, while its accuracy is near zero or one depending on which
    class the evaluator's argmax happens to favour.

    ``evaluator`` is an :class:`~pyhighlights.components.models.spp.base.SPP`
    that selects over words, as an
    :class:`~pyhighlights.components.models.spp.evalx.EvalX` model does.
    ``loader`` yields its batches of the split the frame holds, in any order.
    A selected position beyond a batch's width was cut by truncation, and is
    left out.
    """
    if evaluator.select_over != "word":
        raise ValueError("evaluator readings need an evaluator that selects over words")
    groups = {
        key: {int(row.sample_id): row for row in group.itertuples(index=False)}
        for key, group in frame.groupby(["run", "seed"], sort=True)
    }
    highlights = {
        int(row.sample_id): row.highlights for row in frame.itertuples(index=False)
    }
    tallies = {key: Counter() for key in groups}
    evaluator.eval()
    with th.no_grad():
        for batch in loader:
            ids = [int(sample_id) for sample_id in batch.sample_ids]
            if not any(sample_id in highlights for sample_id in ids):
                continue
            valid = evaluator.selection_valid(batch).float()
            width = valid.shape[1]

            def read(positions: Dict[int, Sequence[int]]) -> th.Tensor:
                mask = th.zeros_like(valid)
                for index, kept in positions.items():
                    mask[index, [p for p in kept if p < width]] = 1.0
                return th.softmax(evaluator.predict(batch, mask), dim=-1)

            on_full = th.softmax(evaluator.predict(batch, valid), dim=-1)
            annotated = {
                index: highlights[sample_id]
                for index, sample_id in enumerate(ids)
                if highlights.get(sample_id) is not None
            }
            on_highlight = read(annotated) if annotated else None
            for key, rows in groups.items():
                present = {
                    index: rows[sample_id].selected
                    for index, sample_id in enumerate(ids)
                    if sample_id in rows
                }
                if not present:
                    continue
                on_selection = read(present)
                tally = tallies[key]
                for index in present:
                    row = rows[ids[index]]
                    tally["documents"] += 1
                    _score(tally, "on_selection", on_selection[index], row.label)
                    tally["agrees"] += (
                        int(on_selection[index].argmax()) == row.predicted
                    )
                    _score(tally, "on_full", on_full[index], row.label)
                    if index in annotated:
                        tally["annotated"] += 1
                        _score(tally, "on_highlight", on_highlight[index], row.label)

    rows = []
    for (run, seed), tally in tallies.items():
        documents, annotated = tally["documents"], tally["annotated"]
        row = {"run": run, "seed": seed, "documents": documents}
        for name in ("on_selection", "p_selection", "agrees", "on_full", "p_full"):
            row[name] = tally[name] / documents if documents else float("nan")
        row["annotated"] = annotated
        for name in ("on_highlight", "p_highlight"):
            row[name] = tally[name] / annotated if annotated else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _score(tally: Counter, name: str, probabilities: th.Tensor, label: int) -> None:
    """Count one reading's accuracy and its probability of the gold label."""
    tally[name] += int(probabilities.argmax()) == label
    tally["p_" + name.removeprefix("on_")] += float(probabilities[label])


def evaluators(
    run: str | Path, split: str = "test"
) -> Iterator[Tuple[str, object, object]]:
    """Every seed of a trained evaluator run, with a loader of ``split``.

    Yields ``(seed, model, loader)``. The run's ``manifest.json`` names the
    task, which is rebuilt to rebuild the model and the corpus it read. Each
    seed directory has to hold exactly one checkpoint, so the run needs
    ``keep_checkpoints``. The registry has to be built.
    """
    run = Path(run)
    manifest = json.loads((run / "manifest.json").read_text())
    task = Registry.from_key(RegistrationKey.parse(registration_key=manifest["key"]))
    splits = task.splits()
    if split not in splits:
        raise KeyError(f"the evaluator's corpus has no {split!r} split")
    loader = task.loaders(splits)[split]
    for directory in sorted(run.glob("seed=*")):
        checkpoints = sorted(directory.glob("*.ckpt"))
        if len(checkpoints) != 1:
            raise ValueError(
                f"{directory} holds {len(checkpoints)} checkpoints, expected one"
            )
        model = task.build_model()
        state = th.load(checkpoints[0], map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"], strict=True)
        yield directory.name.partition("=")[2], model, loader


class SelectionCodeAnalyzer(PredictionAnalyzer):
    """:func:`summarize` over the stored predictions of a results tree.

    A run stores predictions only when its task sets ``store_predictions``.
    ``exclusive`` is passed to :func:`summarize`. :meth:`codes`,
    :meth:`probes`, :meth:`kept` and :meth:`entropy` report the same rows
    through :func:`codes`, :func:`mask_probes`, :func:`kept_by_label` and
    :func:`selection_entropy`. :meth:`evaluated` reads them with a trained
    evaluator through :func:`evaluator_readings`.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        exclusive: bool = False,
        pattern: str = PREDICTIONS,
        split: str = "test",
        latest: bool = True,
    ):
        super().__init__(directory, pattern=pattern, split=split, latest=latest)
        self.exclusive = exclusive

    def analyze(self) -> pd.DataFrame:
        frame = super().analyze()
        return summarize(frame, self.exclusive) if not frame.empty else frame

    def codes(self) -> pd.DataFrame:
        frame = super().analyze()
        return codes(frame) if not frame.empty else frame

    def probes(self) -> pd.DataFrame:
        frame = super().analyze()
        return mask_probes(frame) if not frame.empty else frame

    def kept(self) -> pd.DataFrame:
        frame = super().analyze()
        return kept_by_label(frame) if not frame.empty else frame

    def entropy(self, per_class: bool = True) -> pd.DataFrame:
        frame = super().analyze()
        return selection_entropy(frame, per_class) if not frame.empty else frame

    def evaluated(self, evaluator: str | Path) -> pd.DataFrame:
        """:func:`evaluator_readings` per seed of the ``evaluator`` run.

        One row per run, seed and ``evaluator_seed``, so the spread across
        evaluator seeds can be read. The evaluator has to have read the same
        corpus: a document whose tokens differ from the frame's is refused.
        """
        frame = super().analyze()
        if frame.empty:
            return frame
        readings = []
        for evaluator_seed, model, loader in evaluators(evaluator, self.split):
            if not readings:
                _check_corpus(frame, loader)
            readings.append(
                evaluator_readings(frame, model, loader).assign(
                    evaluator_seed=evaluator_seed
                )
            )
        if not readings:
            raise ValueError(f"{evaluator} holds no evaluator seed")
        return pd.concat(readings, ignore_index=True)


def _check_corpus(frame: pd.DataFrame, loader) -> None:
    """Refuse an evaluator whose corpus places the frame's documents differently."""
    corpus = {
        int(example.sample_id): list(example.tokens)
        for example in loader.dataset.examples
    }
    differing = sum(
        corpus.get(int(row.sample_id)) not in (None, list(row.tokens))
        for row in frame.itertuples(index=False)
    )
    if differing:
        raise ValueError(
            f"{differing} documents read differently in the evaluator's corpus"
        )


def _rivals(frame: pd.DataFrame, exclusive: bool) -> Dict[int, List[Tuple[str, ...]]]:
    """For each class, the other classes' highlights, when ``exclusive``."""
    if not exclusive:
        return {}
    highlights = class_highlights(frame)
    return {
        label: [sequence for other, sequence in highlights.items() if other != label]
        for label in highlights
    }


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else float("nan")


def _probe(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Five-fold cross-validated predictions of a class-balanced probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict

    probe = LogisticRegression(max_iter=3000, class_weight="balanced")
    return cross_val_predict(probe, features, targets, cv=5)


def _entropy(counts: Counter) -> float:
    """Shannon entropy, in bits, of a token count; zero when nothing is counted."""
    total = sum(counts.values())
    if not total:
        return 0.0
    shares = np.array(list(counts.values()), dtype=float) / total
    return float(-(shares * np.log2(shares)).sum())
