from __future__ import annotations

import abc
import csv
import itertools
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from pyhighlights.components.data import HighlightDataset, HighlightExample
from pyhighlights.utility.io import cache_directory, download, extract

COLUMNS = ("sample_id", "text", "tokens", "label", "highlights")

# Priority runs from the split that must stay intact to the one that can
# afford to lose rows. The annotated split comes first: it is the only one
# carrying highlights, so it is the one worth protecting.
PRIORITY = ("test", "val", "train")

R2A_URL = "https://people.csail.mit.edu/yujia/files/r2a/data.zip"
R2A_TASKS = (
    "beer0",
    "beer1",
    "beer2",
    "hotel_Location",
    "hotel_Service",
    "hotel_Cleanliness",
)
R2A_SPLITS = {
    "train": "oracle/{task}.train",
    "val": "oracle/{task}.dev",
    "test": "target/{task}.train",
}


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


def _aligned_highlights(flags: str, width: int) -> List[int]:
    """Parse one rationale, dropping the stray trailing zeros some rows carry.

    Three rows of the R2A release (``hotel_Location`` 59,
    ``hotel_Cleanliness`` 198 and ``beer1`` 119) end with one flag more than
    the text has tokens, and every surplus flag is 0. Trimming those keeps the
    tasks usable. A surplus holding a 1, or a rationale shorter than the text,
    is left alone so the caller reports it: a real misalignment shifts every
    label after the offending position.
    """
    highlights = [int(flag) for flag in flags.split()]
    surplus = highlights[width:]
    return highlights[:width] if surplus and not any(surplus) else highlights


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


def to_examples(frame: pd.DataFrame) -> List[HighlightExample]:
    return [
        HighlightExample(
            sample_id=int(row.sample_id),
            tokens=row.tokens,
            label=int(row.label),
            highlights=row.highlights,
        )
        for row in frame.itertuples()
    ]


class HighlightLoader(abc.ABC):
    """Corpus source: downloads once, hands back one data frame per split.

    Frames carry ``sample_id``, ``text``, ``tokens``, ``label`` and
    ``highlights`` (``None`` where the split has no annotation), with
    ``highlights`` aligned to ``tokens``.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
        remove_leakage: bool = True,
        key: str = "text",
    ):
        self.directory = Path(directory) if directory is not None else cache_directory()
        self.remove_leakage = remove_leakage
        self.key = key
        self.removed: Dict[str, int] = {}
        self._splits: Dict[str, pd.DataFrame] | None = None

    @abc.abstractmethod
    def read(self) -> Dict[str, pd.DataFrame]:
        """Parse the downloaded corpus into one frame per split."""

    def load(self) -> Dict[str, pd.DataFrame]:
        """Splits as data frames, leak-free unless told otherwise.

        Published splits often overlap — see :class:`R2ALoader` — so by
        default the loader hands back repaired ones and records what it
        dropped in :attr:`removed`. Pass ``remove_leakage=False`` to
        reproduce a corpus exactly as distributed.
        """
        if self._splits is None:
            splits = self.read()
            if self.remove_leakage:
                repaired = remove_leakage(splits, key=self.key)
                self.removed = {
                    name: len(splits[name]) - len(repaired[name]) for name in splits
                }
                splits = repaired
            self._splits = splits
        return self._splits

    def leakage(
        self, key: str | None = None, normalize_keys: bool = True
    ) -> pd.DataFrame:
        return leakage(self.load(), key=key or self.key, normalize_keys=normalize_keys)

    def check_leakage(
        self,
        key: str | None = None,
        tolerance: float = 0.0,
        normalize_keys: bool = True,
    ) -> pd.DataFrame:
        """Return the leakage report, raising when a split pair exceeds
        ``tolerance``.

        Meant to be called in a test or before a run: a corpus that shares
        rows between train and test reports highlight scores on examples the
        model was trained on, and nothing downstream can detect that.
        """
        report = self.leakage(key=key, normalize_keys=normalize_keys)
        offending = report[report["ratio"] > tolerance]
        if not offending.empty:
            raise ValueError(
                f"{type(self).__name__} splits share rows above the "
                f"{tolerance} tolerance:\n{offending.to_string(index=False)}"
            )
        return report

    def duplicates(self, key: str | None = None) -> Dict[str, int]:
        return duplicates(self.load(), key=key or self.key)

    def datasets(self) -> Dict[str, HighlightDataset]:
        return {
            name: HighlightDataset(to_examples(frame))
            for name, frame in self.load().items()
        }


class R2ALoader(HighlightLoader):
    """Beer and Hotel aspects from the R2A archive of Bao et al., 2018.

    ``data/target/<task>.train`` is the only file in the release carrying
    per-token annotation, so it is the default ``test`` split despite its
    name — that is the file this line of work reports highlight scores on.

    **The distributed splits overlap.** Every one of the 200 annotated rows of
    each Hotel aspect also appears in that aspect's training file, and Beer
    keeps about two thirds of its validation split inside training. The
    default ``remove_leakage=True`` drops the offending training and
    validation rows, keeping the annotated split whole; ``False`` reproduces
    the release as distributed, leakage included.
    """

    def __init__(
        self,
        task: str = "hotel_Location",
        splits: Mapping[str, str] | None = None,
        url: str = R2A_URL,
        sha256: str | None = None,
        archive_name: str = "r2a.zip",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if task not in R2A_TASKS:
            raise ValueError(f"task must be one of {R2A_TASKS}")
        self.task = task
        self.splits = dict(splits) if splits else dict(R2A_SPLITS)
        self.url = url
        self.sha256 = sha256
        self.archive_name = archive_name

    @property
    def root(self) -> Path:
        return self.directory / "r2a"

    def download(self) -> Path:
        archive = download(
            self.url, self.directory / self.archive_name, sha256=self.sha256
        )
        return extract(archive, self.root)

    def read(self) -> Dict[str, pd.DataFrame]:
        root = self.download()
        return {
            name: self.read_file(root / "data" / pattern.format(task=self.task))
            for name, pattern in self.splits.items()
        }

    @staticmethod
    def read_file(path: Path) -> pd.DataFrame:
        """Read one tab-separated R2A file into the standard columns."""
        source = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
            quoting=csv.QUOTE_NONE,
        )
        labels = pd.to_numeric(source["label"])
        if (labels % 1 != 0).any():
            raise ValueError(
                f"{path} holds continuous labels; pyhighlights expects classes"
            )

        tokens = source["text"].str.split()
        if "rationale" in source.columns:
            highlights = pd.Series(
                [
                    _aligned_highlights(flags, width)
                    for flags, width in zip(source["rationale"], tokens.map(len))
                ],
                dtype=object,
            )
            widths = tokens.map(len) != highlights.map(len)
            if widths.any():
                raise ValueError(
                    f"{path} has rationales misaligned with tokens "
                    f"on {int(widths.sum())} rows"
                )
        else:
            highlights = pd.Series([None] * len(source), dtype=object)

        return pd.DataFrame(
            {
                "sample_id": range(len(source)),
                "text": source["text"],
                "tokens": tokens,
                "label": labels.astype(int),
                "highlights": highlights,
            }
        )


class ToyLoader(HighlightLoader):
    """Synthetic corpus: one trigger phrase per class inside filler tokens.

    Every split is annotated, the highlights are exactly the trigger, and no
    download is involved — which makes it the cheap way to exercise a model,
    a configuration or a training loop end to end.
    """

    def __init__(
        self,
        sizes: Mapping[str, int] | None = None,
        triggers: Sequence[str] = ("a great film", "a dull film"),
        length: int = 24,
        vocabulary_size: int = 32,
        seed: int = 0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if len(triggers) < 2:
            raise ValueError("ToyLoader needs one trigger per class")
        if length < 1 or vocabulary_size < 1:
            raise ValueError("length and vocabulary_size must be positive")
        self.sizes = dict(sizes or {"train": 64, "val": 16, "test": 16})
        self.triggers = list(triggers)
        self.length = length
        self.vocabulary_size = vocabulary_size
        self.seed = seed

    def generate(self, size: int, generator: random.Random) -> pd.DataFrame:
        rows = []
        for index in range(size):
            label = index % len(self.triggers)
            trigger = self.triggers[label].split()
            filler = [
                f"w{generator.randrange(self.vocabulary_size)}"
                for _ in range(self.length)
            ]
            at = generator.randrange(len(filler) + 1)
            tokens = filler[:at] + trigger + filler[at:]
            rows.append(
                {
                    "sample_id": index,
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "label": label,
                    "highlights": [0] * at
                    + [1] * len(trigger)
                    + [0] * (len(filler) - at),
                }
            )
        return pd.DataFrame(rows, columns=list(COLUMNS))

    def read(self) -> Dict[str, pd.DataFrame]:
        return {
            name: self.generate(size, random.Random(f"{self.seed}-{name}"))
            for name, size in self.sizes.items()
        }


class HateXplainLoader(HighlightLoader):
    """Hate-speech posts with per-annotator token rationales.

    From Mathew et al., 2021, *HateXplain: A Benchmark Dataset for Explainable
    Hate Speech Detection*. Three annotators label every post and mark the
    tokens supporting a non-``normal`` label, so both the label and the
    highlights are aggregated:

    - ``label``: majority vote. Posts where all three annotators disagree —
      919 of 20148 — have no majority; ``ties="drop"`` removes them, as the
      paper does, and ``ties="keep"`` resolves them by annotator order.
    - ``highlights``: ``"majority"`` keeps a token marked by more than half of
      the rationale vectors, ``"union"`` by any, ``"intersection"`` by all.

    ``normal`` posts carry no rationale by design, and neither do 580
    non-normal ones; both come back as all-zero highlights, meaning "no token
    was marked" rather than "not annotated".
    """

    URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    DIVISIONS_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
    LABELS = ("hatespeech", "normal", "offensive")

    def __init__(
        self,
        url: str = URL,
        divisions_url: str = DIVISIONS_URL,
        labels: Sequence[str] = LABELS,
        rationale: str = "majority",
        ties: str = "drop",
        **kwargs,
    ):
        super().__init__(**kwargs)
        if rationale not in ("majority", "union", "intersection"):
            raise ValueError("rationale must be majority, union or intersection")
        if ties not in ("drop", "keep"):
            raise ValueError("ties must be drop or keep")
        self.url = url
        self.divisions_url = divisions_url
        self.labels = {name: index for index, name in enumerate(labels)}
        self.rationale = rationale
        self.ties = ties

    def download(self) -> Dict[str, Path]:
        root = self.directory / "hatexplain"
        return {
            "posts": download(self.url, root / "dataset.json"),
            "divisions": download(self.divisions_url, root / "post_id_divisions.json"),
        }

    def aggregate(self, vectors: List[List[int]], width: int) -> List[int]:
        valid = [vector for vector in vectors if len(vector) == width]
        if not valid:
            return [0] * width
        counts = [sum(flags) for flags in zip(*valid)]
        if self.rationale == "union":
            return [int(count > 0) for count in counts]
        if self.rationale == "intersection":
            return [int(count == len(valid)) for count in counts]
        return [int(count * 2 > len(valid)) for count in counts]

    def label(self, annotators: List[dict]) -> int | None:
        votes = Counter(annotator["label"] for annotator in annotators)
        (name, count), *rest = votes.most_common()
        if count == 1 and self.ties == "drop":
            return None
        if name not in self.labels:
            raise ValueError(f"unexpected HateXplain label {name}")
        return self.labels[name]

    def read(self) -> Dict[str, pd.DataFrame]:
        paths = self.download()
        posts = json.loads(paths["posts"].read_text())
        divisions = json.loads(paths["divisions"].read_text())

        splits = {}
        for name, post_ids in divisions.items():
            rows = []
            for post_id in post_ids:
                post = posts[post_id]
                label = self.label(post["annotators"])
                if label is None:
                    continue
                tokens = list(post["post_tokens"])
                rows.append(
                    {
                        "sample_id": len(rows),
                        "text": " ".join(tokens),
                        "tokens": tokens,
                        "label": label,
                        "highlights": self.aggregate(post["rationales"], len(tokens)),
                    }
                )
            splits[name] = pd.DataFrame(rows, columns=list(COLUMNS))
        order = ("train", "val", "test")
        return {name: splits[name] for name in order if name in splits}


class ERASERLoader(HighlightLoader):
    """Document classification with evidence spans, from the ERASER benchmark.

    DeYoung et al., 2020, *ERASER: A Benchmark to Evaluate Rationalized NLP
    Models*. A task ships a ``docs`` directory of whitespace-tokenized
    documents and one JSONL file per split whose rows carry a
    ``classification`` and ``evidences`` — groups of ``[start_token,
    end_token)`` spans into the document. Those spans become the highlights.

    Only single-document tasks fit the select-then-predict input, which takes
    one token sequence and no query. ``movies`` is such a task; the
    query-based ones would need the question folded into the document, which
    would change what a model is shown without saying so, and are refused
    until pyhighlights has a place to put a query.
    """

    URL = "https://www.eraserbenchmark.com/zipped/{task}.tar.gz"
    TASKS = ("movies",)
    QUERY_TASKS = (
        "boolq",
        "esnli",
        "evidence_inference",
        "fever",
        "multirc",
        "scifact",
    )
    SPLITS = {"train": "train.jsonl", "val": "val.jsonl", "test": "test.jsonl"}

    def __init__(
        self,
        task: str = "movies",
        splits: Mapping[str, str] | None = None,
        url: str = URL,
        sha256: str | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if task in self.QUERY_TASKS:
            raise ValueError(
                f"{task} pairs a query with its document; pyhighlights models "
                "take a single token sequence, so only "
                f"{self.TASKS} are supported"
            )
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}")
        self.task = task
        self.splits = dict(splits) if splits else dict(self.SPLITS)
        self.url = url
        self.sha256 = sha256

    @property
    def root(self) -> Path:
        return self.directory / "eraser" / self.task

    def download(self) -> Path:
        url = self.url.format(task=self.task)
        archive = download(
            url, self.directory / f"eraser-{self.task}.tar.gz", sha256=self.sha256
        )
        return extract(archive, self.root)

    @staticmethod
    def document_id(row: dict) -> str:
        for group in row.get("evidences") or []:
            for evidence in group:
                if evidence.get("docid"):
                    return evidence["docid"]
        docids = row.get("docids")
        return docids[0] if docids else row["annotation_id"]

    def read_file(self, path: Path, documents: Path, labels: Dict[str, int]):
        rows = []
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            tokens = (documents / self.document_id(row)).read_text().split()
            highlights = [0] * len(tokens)
            for group in row.get("evidences") or []:
                for evidence in group:
                    start = int(evidence["start_token"])
                    end = int(evidence["end_token"])
                    if not 0 <= start <= end <= len(tokens):
                        raise ValueError(
                            f"{path} has an evidence span outside "
                            f"{self.document_id(row)}"
                        )
                    highlights[start:end] = [1] * (end - start)
            rows.append(
                {
                    "sample_id": len(rows),
                    "text": " ".join(tokens),
                    "tokens": tokens,
                    "label": labels[row["classification"]],
                    "highlights": highlights,
                }
            )
        return pd.DataFrame(rows, columns=list(COLUMNS))

    def read(self) -> Dict[str, pd.DataFrame]:
        root = self.download() / self.task
        documents = root / "docs"
        classifications = sorted(
            {
                json.loads(line)["classification"]
                for name in self.splits.values()
                for line in (root / name).read_text().splitlines()
                if line.strip()
            }
        )
        labels = {name: index for index, name in enumerate(classifications)}
        return {
            name: self.read_file(root / path, documents, labels)
            for name, path in self.splits.items()
        }
