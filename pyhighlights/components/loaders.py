from __future__ import annotations

import abc
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from pyhighlights.components.data import (
    COLUMNS,
    HighlightDataset,
    HighlightExample,
)
from pyhighlights.utility.io import cache_directory, download, extract

R2A_URL = "https://people.csail.mit.edu/yujia/files/r2a/data.zip"
BEER_TASKS = ("beer0", "beer1", "beer2")
HOTEL_TASKS = ("hotel_Location", "hotel_Service", "hotel_Cleanliness")
R2A_TASKS = BEER_TASKS + HOTEL_TASKS
R2A_SPLITS = {
    "train": "oracle/{task}.train",
    "val": "oracle/{task}.dev",
    "test": "target/{task}.train",
}


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


def to_examples(frame: pd.DataFrame) -> List[HighlightExample]:
    if frame["label"].isna().any():
        raise ValueError(
            "rows carry no label: this corpus is annotated by several people "
            "and the judgements have not been reduced yet. Run an "
            "AnnotationAggregator over the splits first."
        )
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

    Frames carry :data:`~pyhighlights.components.data.COLUMNS`, with
    ``highlights`` aligned to ``tokens`` and ``None`` where a split has no
    annotation. The corpus comes back **as distributed** -- overlapping
    splits, per-annotator judgements and all. Repairing or reducing it is
    :mod:`~pyhighlights.components.preprocessors`, and what a study does there
    is its own decision; the loader that fetched the files has no business
    making it, and splits the user built themselves deserve the same choices.
    """

    def __init__(
        self,
        directory: str | Path | None = None,
    ):
        self.directory = Path(directory) if directory is not None else cache_directory()
        self._splits: Dict[str, pd.DataFrame] | None = None

    @abc.abstractmethod
    def read(self) -> Dict[str, pd.DataFrame]:
        """Parse the downloaded corpus into one frame per split."""

    def load(self) -> Dict[str, pd.DataFrame]:
        """The splits as distributed, parsed once and kept."""
        if self._splits is None:
            self._splits = self.read()
        return self._splits

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

    TASKS = R2A_TASKS

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
        if task not in self.TASKS:
            raise ValueError(f"task must be one of {self.TASKS}")
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


class BeerLoader(R2ALoader):
    """The three Beer aspects of the R2A archive: appearance, aroma, palate.

    A corpus of its own rather than a ``task`` string, so the aspect is the
    only thing left to choose and the artefact each aspect is fetched from has
    somewhere to live.
    """

    TASKS = BEER_TASKS

    def __init__(self, task: str = "beer0", **kwargs):
        super().__init__(task=task, **kwargs)


class HotelLoader(R2ALoader):
    """The three Hotel aspects of the R2A archive: location, service, cleanliness."""

    TASKS = HOTEL_TASKS

    def __init__(self, task: str = "hotel_Location", **kwargs):
        super().__init__(task=task, **kwargs)


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
    tokens supporting a non-``normal`` label.

    **Every judgement is kept.** ``label`` and ``highlights`` come back unset,
    and the raw material sits in ``annotator_labels`` and
    ``annotator_highlights``; how three annotators become one label and one
    highlight vector is a choice
    :class:`~pyhighlights.components.preprocessors.AnnotationAggregator`
    makes, and 919 of the 20148 posts have no majority at all. Until it has
    run, the splits are not yet examples and
    :meth:`~HighlightLoader.datasets` says so.
    """

    URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/dataset.json"
    DIVISIONS_URL = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/post_id_divisions.json"
    LABELS = ("hatespeech", "normal", "offensive")

    def __init__(
        self,
        url: str = URL,
        divisions_url: str = DIVISIONS_URL,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.divisions_url = divisions_url

    def download(self) -> Dict[str, Path]:
        root = self.directory / "hatexplain"
        return {
            "posts": download(self.url, root / "dataset.json"),
            "divisions": download(self.divisions_url, root / "post_id_divisions.json"),
        }

    def read(self) -> Dict[str, pd.DataFrame]:
        paths = self.download()
        posts = json.loads(paths["posts"].read_text())
        divisions = json.loads(paths["divisions"].read_text())

        splits = {}
        for name, post_ids in divisions.items():
            rows = []
            for post_id in post_ids:
                post = posts[post_id]
                tokens = list(post["post_tokens"])
                rows.append(
                    {
                        "sample_id": len(rows),
                        "text": " ".join(tokens),
                        "tokens": tokens,
                        "label": None,
                        "highlights": None,
                        "annotator_labels": [
                            annotator["label"] for annotator in post["annotators"]
                        ],
                        "annotator_highlights": [
                            list(vector) for vector in post["rationales"]
                        ],
                    }
                )
            splits[name] = pd.DataFrame(
                rows,
                columns=[*COLUMNS, "annotator_labels", "annotator_highlights"],
            )
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


class MoviesLoader(ERASERLoader):
    """The ERASER ``movies`` task: sentiment with evidence spans.

    The only single-document ERASER task, and the one this line of work
    reports on.
    """

    def __init__(self, task: str = "movies", **kwargs):
        super().__init__(task=task, **kwargs)
