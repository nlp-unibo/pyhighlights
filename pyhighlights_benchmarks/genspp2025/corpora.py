"""The paper's synthetic corpus, as released rather than as regenerated."""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pyhighlights.components.data import COLUMNS
from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.utility.io import download, extract

__all__ = ["GenSPPToyLoader"]


class GenSPPToyLoader(HighlightLoader):
    """The 10,000 sequences the GenSPP paper trained on.

    Each row is a twenty-character string over the lowercase alphabet with a
    three-character pattern hidden in it; the class is which pattern, and the
    highlight is exactly where it sits. **Tokens are characters**, so the
    vocabulary is the alphabet and an embedding table of 26 rows covers it.

    :class:`~pyhighlights.components.loaders.ToyLoader` generates a corpus of
    the same shape but not this one, and a reproduction needs this one: the
    numbers in the paper are for these ten thousand sequences.

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
