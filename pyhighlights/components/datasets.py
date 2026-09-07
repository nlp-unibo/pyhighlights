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
from typing import Dict, List, Mapping, Sequence

import pandas as pd

from pyhighlights.components.data import HighlightDataset, HighlightExample

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
