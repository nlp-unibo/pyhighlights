from __future__ import annotations

import abc
import csv
import hashlib
import itertools
import os
import re
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping

import pandas as pd

from pyhighlights.components.data import HighlightDataset, HighlightExample

COLUMNS = ("sample_id", "text", "tokens", "label", "highlights")

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


def cache_directory() -> Path:
    """Shared download cache, overridable with ``PYHIGHLIGHTS_CACHE``."""
    default = Path.home() / ".cache" / "pyhighlights"
    return Path(os.environ.get("PYHIGHLIGHTS_CACHE") or default)


def download(url: str, target: Path, sha256: str | None = None) -> Path:
    """Fetch ``url`` into ``target`` unless already there; verify if asked."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_name(target.name + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(target)

    if sha256 is not None:
        digest = hashlib.sha256()
        with target.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        if digest.hexdigest() != sha256:
            raise ValueError(f"{target} does not match the expected sha256")
    return target


def _checked_names(names: List[str]) -> List[str]:
    for name in names:
        parts = Path(name).parts
        if Path(name).is_absolute() or ".." in parts:
            raise ValueError(f"refusing to extract unsafe archive path {name}")
    return names


def extract(archive: Path, directory: Path) -> Path:
    """Unpack ``archive`` into ``directory`` once; return ``directory``."""
    if directory.exists():
        return directory

    staging = directory.with_name(directory.name + ".partial")
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            source.extractall(staging, members=_checked_names(source.namelist()))
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as source:
            _checked_names(source.getnames())
            source.extractall(staging)
    else:
        raise ValueError(f"{archive} is neither a zip nor a tar archive")
    staging.replace(directory)
    return directory


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

    def __init__(self, directory: str | Path | None = None):
        self.directory = Path(directory) if directory is not None else cache_directory()
        self._splits: Dict[str, pd.DataFrame] | None = None

    @abc.abstractmethod
    def read(self) -> Dict[str, pd.DataFrame]:
        """Parse the downloaded corpus into one frame per split."""

    def load(self) -> Dict[str, pd.DataFrame]:
        if self._splits is None:
            self._splits = self.read()
        return self._splits

    def leakage(self, key: str = "text", normalize_keys: bool = True) -> pd.DataFrame:
        return leakage(self.load(), key=key, normalize_keys=normalize_keys)

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
    Its rows also appear in the training file, completely so for every Hotel
    aspect, which :meth:`leakage` quantifies.
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
