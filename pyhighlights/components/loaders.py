from __future__ import annotations

import abc
import csv
import json
import random
import string
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

from pyhighlights.components.data import (
    COLUMNS,
    HighlightDataset,
    HighlightExample,
)
from pyhighlights.utility.io import cache_directory, download, extract

R2A_URL = "https://people.csail.mit.edu/yujia/files/r2a/data.zip"
#: The release the split manifests on Zenodo were built against. Pinned so a
#: reproduction fails loudly on a changed upstream rather than training on it.
R2A_SHA256 = "23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de"
BEER_TASKS = ("beer0", "beer1", "beer2")
HOTEL_TASKS = ("hotel_Location", "hotel_Service", "hotel_Cleanliness")
R2A_TASKS = BEER_TASKS + HOTEL_TASKS
R2A_SPLITS = {
    "train": "oracle/{task}.train",
    "val": "oracle/{task}.dev",
    "test": "target/{task}.train",
}


def _aligned_highlights(flags: str, width: int) -> List[int]:
    """Parse one highlight, dropping the stray trailing zeros some rows carry.

    Three rows of the R2A release (``hotel_Location`` 59,
    ``hotel_Cleanliness`` 198 and ``beer1`` 119) end with one flag more than
    the text has tokens, and every surplus flag is 0. Trimming those keeps the
    tasks usable. A surplus holding a 1, or a highlight shorter than the text,
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
    # `knowledge` is an extra column rather than one of COLUMNS: a corpus with
    # a knowledge base is the exception, and every loader that has none should
    # keep returning exactly the frame it returns today.
    links = frame["knowledge"] if "knowledge" in frame else None
    return [
        HighlightExample(
            sample_id=int(row.sample_id),
            tokens=row.tokens,
            label=int(row.label),
            highlights=row.highlights,
            knowledge=None if links is None else links.iloc[position],
        )
        for position, row in enumerate(frame.itertuples())
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

    def knowledge(self) -> Sequence[Sequence[str]] | None:
        """The corpus's knowledge base, one entry as a list of tokens.

        Optional, like :meth:`SPPBackbone.load_embeddings`: most corpora have
        none and say so by inheriting this. A corpus that has one returns the
        entries **in the order its annotation indexes them**, because the
        links a split carries are positions into this sequence and nothing
        downstream can check an order it was never told.

        It is a property of the corpus rather than of a sample: every example
        of a run shares it, so it is loaded once and never collated into a
        batch.
        """
        return None

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

    The repaired splits are published as manifests, so a reader can check a run
    against the rows it should have seen rather than take the repair on trust:
    Beer at `10.5281/zenodo.22703544
    <https://doi.org/10.5281/zenodo.22703544>`_ and Hotel at
    `10.5281/zenodo.22711382 <https://doi.org/10.5281/zenodo.22711382>`_. They
    are a receipt rather than an input -- ``sha256`` pins the upstream archive
    and the repair is deterministic, so the splits come out the same without
    fetching either.
    """

    TASKS = R2A_TASKS

    def __init__(
        self,
        task: str = "hotel_Location",
        splits: Mapping[str, str] | None = None,
        url: str = R2A_URL,
        sha256: str | None = R2A_SHA256,
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
    """Synthetic corpus: each class is a set of character patterns to be found.

    **Tokens are characters**, as they are in every toy corpus of this line of
    work: ``GenSPPToyLoader`` reads a released one with ``list(row.text)``, and
    the generator it comes from samples an alphabet. A trigger is therefore a
    string of characters like ``"aa"``, not a phrase.

    A class's trigger is a **conjunction**: every pattern in it has to appear
    for the class to hold. One pattern is the short spelling of a conjunction
    of one, so both of these are triggers::

        triggers = ["aa", "bc"]                           # a pattern per class
        triggers = [["aba", "baa"], ["baa", "abb"]]       # two, both required

    The second form is the one worth having. When no pattern belongs to a
    single class -- ``baa`` sits in both classes above -- no single n-gram
    identifies a class, and a model that memorises one cannot pass. It also
    makes the gold highlight **several disjoint spans** rather than one run,
    which is the shape a real highlight has.

    Patterns **overwrite** filler at disjoint positions, in a random order and
    with at least one filler character between them. Overwriting rather than
    inserting is what keeps every document exactly ``length`` tokens long: a
    class whose patterns are longer would otherwise produce longer documents,
    and the document's own length would say which class it is without reading
    a character of it. The released generator overwrites for the same reason.
    A generated sample must satisfy its own class and no other; one that does
    not is drawn again.

    Every split is annotated, the highlights are exactly the patterns, and no
    download is involved -- which makes it the cheap way to exercise a model, a
    configuration or a training loop end to end.

    ``contaminations`` scatters proper chunks of the patterns through the
    filler, which is what stops a fragment from being enough to classify:
    without it ``bc`` occurs nowhere but inside ``abc``, so detecting ``bc``
    names that class exactly as well as detecting ``abc`` does. Off by default,
    since the registered corpus is a smoke test and a cheap one is the point. A
    reproduction of the published numbers reads the released pickle through
    :class:`GenSPPToyLoader`.

    Whatever the settings,
    :class:`~pyhighlights.components.shortcuts.ShortcutDetector` is what says
    the corpus is a control: it removes the annotated positions and requires
    that nothing left predicts the label.
    """

    #: Where filler characters come from, minus whatever the triggers use.
    ALPHABET = string.ascii_lowercase
    #: Draws allowed per sample before the placement is called impossible.
    ATTEMPTS = 100

    def __init__(
        self,
        sizes: Mapping[str, int] | None = None,
        triggers: Sequence[str | Sequence[str]] = ("aa", "bc"),
        length: int = 20,
        vocabulary_size: int = 20,
        contaminations: int = 0,
        min_chunk: int = 2,
        seed: int = 0,
        **kwargs,
    ):
        """``length`` is the document's length, patterns included.

        ``vocabulary_size`` is how many filler characters there are. They are
        drawn from the letters no trigger uses, so a pattern can only
        appear where this put one: two adjacent filler characters can never
        spell ``"aa"`` if ``a`` is not a filler character. The released
        generator reaches the same end by cleaning the sequence and rejecting
        any sample that accidentally satisfies another class.
        """
        super().__init__(**kwargs)
        if len(triggers) < 2:
            raise ValueError("ToyLoader needs one trigger per class")
        self.triggers = [
            (trigger,) if isinstance(trigger, str) else tuple(trigger)
            for trigger in triggers
        ]
        if length < 1 or vocabulary_size < 1:
            raise ValueError("length and vocabulary_size must be positive")
        if any(not pattern for trigger in self.triggers for pattern in trigger):
            raise ValueError("a trigger is at least one character")
        if any(not trigger for trigger in self.triggers):
            raise ValueError("a trigger is at least one pattern")
        # Every pattern, plus one filler character between consecutive ones.
        widest = max(
            sum(len(pattern) for pattern in trigger) + len(trigger) - 1
            for trigger in self.triggers
        )
        if widest > length:
            raise ValueError(
                f"a class needs {widest} tokens for its patterns and the gaps "
                f"between them, and length is {length}"
            )
        used = {
            character
            for trigger in self.triggers
            for pattern in trigger
            for character in pattern
        }
        self.alphabet = [letter for letter in self.ALPHABET if letter not in used]
        if len(self.alphabet) < vocabulary_size:
            raise ValueError(
                f"{vocabulary_size} filler characters were asked for and the "
                f"triggers leave {len(self.alphabet)} of {len(self.ALPHABET)}"
            )
        self.alphabet = self.alphabet[:vocabulary_size]
        if contaminations < 0 or min_chunk < 1:
            raise ValueError("contaminations and min_chunk cannot be negative")
        self.sizes = dict(sizes or {"train": 64, "val": 16, "test": 16})
        self.length = length
        self.vocabulary_size = vocabulary_size
        self.contaminations = contaminations
        self.min_chunk = min_chunk
        self.seed = seed
        # A chunk is a *proper* piece of a pattern -- strictly shorter, so it
        # can never satisfy the class it was cut from. One that spells another
        # class's pattern outright is dropped: every inserted run sits alone
        # between filler characters, so a run that is a pattern *is* that
        # pattern being present, and the sample would carry a class nobody
        # annotated.
        self.chunks = [
            chunk
            for trigger in self.triggers
            for pattern in trigger
            for width in range(min_chunk, len(pattern))
            for start in range(len(pattern) - width + 1)
            if not self.satisfied(chunk := pattern[start : start + width])
        ]
        if contaminations and not self.chunks:
            raise ValueError(
                f"no pattern is longer than min_chunk={min_chunk}, so there is "
                f"nothing to contaminate with"
            )
        if contaminations:
            widest += contaminations * (max(len(chunk) for chunk in self.chunks) + 1)
            if widest > length:
                raise ValueError(
                    f"a class needs {widest} tokens for its patterns, "
                    f"{contaminations} contaminations and the gaps between "
                    f"them, and length is {length}"
                )

    def satisfied(self, text: str) -> set:
        """Which classes' conjunctions ``text`` holds, as a set of labels.

        A valid sample satisfies exactly its own. Public because it is what the
        corpus means -- a shortcut scan asks it about texts this never wrote.
        """
        return {
            label
            for label, trigger in enumerate(self.triggers)
            if all(pattern in text for pattern in trigger)
        }

    def place(self, trigger, generator: random.Random):
        """One sample's tokens and highlight, from a class's patterns.

        The patterns overwrite filler at disjoint positions, in a random order:
        disjoint with a gap so that each is its own span, random so that the
        order they were listed in is not a feature.

        The positions are drawn uniformly over the arrangements that fit. With
        widths ``w`` and ``k`` runs, ``length - sum(w) - (k - 1)`` tokens of
        filler are free to sit in the ``k + 1`` gaps; choosing ``k`` cuts out
        of ``slack + k`` picks one such arrangement, and each is equally
        likely. Sampling each start independently and rejecting the overlaps
        would not be uniform, and where a pattern sits is a channel this corpus
        exists to keep shut.

        **Contaminating chunks are placed in the same draw as the patterns.**
        Placing them afterwards, or keeping only the samples that came out
        valid, conditions the arrangement on the class: reject the draws where
        a chunk completes a second copy of the pattern and the characters
        *beside* the highlight stop being class-independent -- which survives
        removing the highlight and is a shortcut. Here nothing is rejected,
        because a filler character separates every run and no chunk spells a
        pattern, so no arrangement can be invalid.
        """
        tokens = [generator.choice(self.alphabet) for _ in range(self.length)]
        highlights = [0] * self.length

        runs = [(pattern, 1) for pattern in trigger]
        runs += [(generator.choice(self.chunks), 0) for _ in range(self.contaminations)]
        generator.shuffle(runs)
        widths = [len(pattern) for pattern, _ in runs]
        slack = self.length - sum(widths) - (len(runs) - 1)
        cuts = sorted(generator.sample(range(slack + len(runs)), len(runs)))

        for index, (cut, (pattern, gold)) in enumerate(zip(cuts, runs)):
            start = cut + sum(widths[:index])
            tokens[start : start + len(pattern)] = list(pattern)
            if gold:
                highlights[start : start + len(pattern)] = [1] * len(pattern)
        return tokens, highlights

    def generate(self, size: int, generator: random.Random) -> pd.DataFrame:
        rows = []
        for index in range(size):
            label = index % len(self.triggers)
            for _ in range(self.ATTEMPTS):
                tokens, highlights = self.place(self.triggers[label], generator)
                text = "".join(tokens)
                # Placement cannot spell a pattern out of filler, but it can
                # satisfy a second class outright when one class's patterns are
                # a subset of another's, and contamination can spell one out of
                # a chunk and the filler beside it. Both are corpora nobody can
                # label, so the draw is repeated rather than recorded.
                if self.satisfied(text) != {label}:
                    continue
                # The highlight also has to be the whole truth: a pattern the
                # contamination reproduced elsewhere would sit unmarked.
                marked = "".join(
                    token for token, flag in zip(tokens, highlights) if flag
                )
                if all(
                    text.count(pattern) == marked.count(pattern)
                    for pattern in self.triggers[label]
                ):
                    break
            else:
                raise ValueError(
                    f"class {label} could not be placed without satisfying "
                    f"another or repeating a pattern in {self.ATTEMPTS} "
                    f"attempts; its patterns {self.triggers[label]} may cover "
                    f"another class's"
                )
            rows.append(
                {
                    "sample_id": index,
                    "text": text,
                    "tokens": tokens,
                    "label": label,
                    "highlights": highlights,
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

    #: The commit both files are read at. A branch name is not a version: the
    #: same key would name different rows after an upstream push, and a run
    #: made before it could not be told from a run made after. The benchmark
    #: publishes no digest of its own, so :attr:`SHA256` and
    #: :attr:`DIVISIONS_SHA256` were computed against this commit -- which is
    #: what ``master`` resolved to as of 2026-09-15, byte for byte.
    COMMIT = "01d742279dac941981f53806154481c0e15ee686"
    URL = (
        "https://raw.githubusercontent.com/hate-alert/HateXplain/"
        f"{COMMIT}/Data/dataset.json"
    )
    #: Digest of the 12256170-byte ``dataset.json`` that commit holds.
    SHA256 = "63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb"
    DIVISIONS_URL = (
        "https://raw.githubusercontent.com/hate-alert/HateXplain/"
        f"{COMMIT}/Data/post_id_divisions.json"
    )
    #: Digest of the official split map. Separate from :attr:`SHA256` because
    #: they are separate downloads: one can change without the other.
    DIVISIONS_SHA256 = (
        "c2fb0d89862e7897b11ea3e9380753f15a793482b4b70ad0532dfb1212212835"
    )
    LABELS = ("hatespeech", "normal", "offensive")

    def __init__(
        self,
        url: str = URL,
        divisions_url: str = DIVISIONS_URL,
        sha256: str | None = SHA256,
        divisions_sha256: str | None = DIVISIONS_SHA256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.url = url
        self.divisions_url = divisions_url
        self.sha256 = sha256
        self.divisions_sha256 = divisions_sha256

    def download(self) -> Dict[str, Path]:
        root = self.directory / "hatexplain"
        return {
            "posts": download(self.url, root / "dataset.json", sha256=self.sha256),
            "divisions": download(
                self.divisions_url,
                root / "post_id_divisions.json",
                sha256=self.divisions_sha256,
            ),
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
    #: The archive the ``movies`` split manifest was built against. The
    #: benchmark publishes no digest of its own. One constant serves because
    #: :attr:`TASKS` is one task; a second would want this keyed by task.
    SHA256 = "66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee"
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
        sha256: str | None = SHA256,
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


class GenSPPToyLoader(HighlightLoader):
    """The 10,000 sequences the GenSPP paper trained on.

    Each row is a twenty-character string over the lowercase alphabet with a
    three-character pattern hidden in it; the class is which pattern, and the
    highlight is exactly where it sits. **Tokens are characters**, so the
    vocabulary is the alphabet and an embedding table of 26 rows covers it.

    :class:`ToyLoader` generates a corpus of the same shape and not this one.
    The two are not alternatives: one samples an alphabet, this one reads ten
    thousand released sequences, and only these are the rows the paper's
    numbers are for. Nothing here can be regenerated -- with ``url=None`` it
    refuses rather than synthesising a corpus that would look right and be
    different.

    Published at `10.5281/zenodo.22711449
    <https://doi.org/10.5281/zenodo.22711449>`_ under CC-BY-4.0 by both authors
    of the paper, so the loader fetches it rather than being handed it.

    Splits follow the released baselines -- the first 80% train, the rest
    test, and a fifth of train sampled off for validation, drawn from
    ``split_seed`` because the released script seeds everything at 15000
    before sampling.
    """

    #: The published artifact, `10.5281/zenodo.22711449
    #: <https://doi.org/10.5281/zenodo.22711449>`_. The version record rather
    #: than the concept one, because :attr:`SHA256` pins these exact bytes.
    URL: str | None = (
        "https://zenodo.org/api/records/22711449/files/"
        "pyhighlights-genspp-toy-v1.zip/content"
    )
    #: Digest of the artifact :attr:`URL` names.
    SHA256 = "5b0886163b215b932b242ce4910cd8d60b46fa79cfdfdde41e9646d99d9ebc92"
    #: The corpus inside that archive.
    MEMBER = "toy_dataset.pkl"

    def __init__(
        self,
        url: str | None = URL,
        sha256: str | None = SHA256,
        member: str = MEMBER,
        archive_name: str = "pyhighlights-genspp-toy-v1.zip",
        train_ratio: float = 0.8,
        val_ratio: float = 0.2,
        split_seed: int = 15000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if not 0.0 < train_ratio < 1.0:
            raise ValueError("train_ratio must be between zero and one")
        if not 0.0 <= val_ratio < 1.0:
            raise ValueError("val_ratio must be between zero and one")
        self.url = url
        self.sha256 = sha256
        self.member = member
        self.archive_name = archive_name
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed

    def download(self) -> Path:
        """The corpus pickle, fetching and unpacking the artifact if needed.

        ``url`` may be the published archive, a local copy of it, or a local
        ``toy_dataset.pkl`` -- the Zenodo record holds the artifact rather than
        a loose pickle, and the artifact is what carries the manifest, the
        licence and the citation alongside the data.
        """
        if self.url is None:
            raise ValueError(
                "the GenSPP toy corpus has no download URL: pass url= with the "
                "published artifact, or point it at a local toy_dataset.pkl"
            )
        root = self.directory / "genspp2025"
        source = Path(self.url)
        if not source.is_file():
            source = download(self.url, root / self.archive_name, sha256=self.sha256)
        if not zipfile.is_zipfile(source):
            return source
        return extract(source, root / "toy") / self.member

    def read(self) -> Dict[str, pd.DataFrame]:
        frame = pd.read_pickle(self.download())
        rows = []
        for index, row in enumerate(frame.itertuples()):
            tokens = list(row.text)
            highlights = [0] * len(tokens)
            for position in row.structure_indexes:
                highlights[position] = 1
            rows.append(
                {
                    "sample_id": index,
                    "text": row.text,
                    "tokens": tokens,
                    "label": int(row.label),
                    "highlights": highlights,
                }
            )
        corpus = pd.DataFrame(rows, columns=list(COLUMNS))

        train_count = int(len(corpus) * self.train_ratio)
        train, test = corpus[:train_count], corpus[train_count:]
        val = train.sample(
            n=int(train_count * self.val_ratio),
            random_state=np.random.RandomState(self.split_seed),
        )
        train = train[~train.index.isin(val.index)]
        return {
            name: frame.reset_index(drop=True)
            for name, frame in (("train", train), ("val", val), ("test", test))
        }
