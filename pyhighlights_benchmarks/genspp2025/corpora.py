"""The paper's synthetic corpus, as released rather than as regenerated."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from pyhighlights.components.data import COLUMNS
from pyhighlights.components.loaders import HighlightLoader
from pyhighlights.utility.io import download

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

    Splits follow the released baselines -- the first 80% train, the rest
    test, and a fifth of train sampled off for validation, drawn from
    ``split_seed`` because the released script seeds everything at 15000
    before sampling.
    """

    #: Where the corpus will be published. Until it is, ``url`` has to be given.
    URL: str | None = None

    def __init__(
        self,
        url: str | None = URL,
        sha256: str | None = None,
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
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.split_seed = split_seed

    def download(self) -> Path:
        if self.url is None:
            raise ValueError(
                "the GenSPP toy corpus has no download URL yet: pass url= with "
                "the published artifact, or point it at a local toy_dataset.pkl"
            )
        source = Path(self.url)
        if source.is_file():
            return source
        return download(
            self.url,
            self.directory / "genspp2025" / "toy_dataset.pkl",
            sha256=self.sha256,
        )

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
