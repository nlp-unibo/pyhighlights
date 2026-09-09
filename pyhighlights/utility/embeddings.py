"""Pretrained token vectors, read off disk into a vocabulary and a matrix.

A backbone learns its embedding table by default. Reproducing a published
result usually means not learning it: the vectors are fixed, frozen, and the
vocabulary is whatever the release covers. That is a corpus-and-file question
rather than a model one, so it lives here and reaches the model as a tensor.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Tuple

import torch as th

__all__ = ["load_vectors"]


def load_vectors(
    path: str | Path,
    tokens: Iterable[str] | None = None,
    pretrained_only: bool = True,
    generator: th.Generator | None = None,
) -> Tuple[Dict[str, int], th.Tensor]:
    """Read a GloVe-style text file into a vocabulary and its embedding matrix.

    The file is one line per token: the token, then its vector, whitespace
    separated -- the format GloVe, fastText and word2vec's text export share.

    ``tokens`` restricts the result to a corpus, and should be the training
    vocabulary: keeping vectors for words only the test split uses costs
    memory and tells the model which words those are.

    ``pretrained_only`` decides what happens to a corpus token the file has no
    vector for. ``True`` drops it, so every row is a released vector and the
    unknown id absorbs the rest -- what a reproduction usually wants, since a
    randomly initialised row inside a frozen table is noise nothing can learn
    away. ``False`` keeps the token and gives it a random row.

    Row ``0`` is zeros and belongs to the unknown and padding id, matching
    :class:`~pyhighlights.components.data.VocabularyTokenizer`, so the ids the
    vocabulary hands out start at ``1``.
    """
    wanted = None if tokens is None else set(tokens)
    vocabulary: Dict[str, int] = {}
    vectors: list[th.Tensor] = []

    # utf-8-sig: released vector files are sometimes shipped with a byte order
    # mark, and it would otherwise ride along on the first token, which then
    # matches nothing in the corpus.
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for index, line in enumerate(handle):
            token, _, values = line.rstrip("\n").partition(" ")
            # word2vec's text export opens with a "<rows> <columns>" header,
            # which would otherwise be read as a one-dimensional vector for a
            # token spelled like a number.
            if index == 0 and token.isdigit() and values.strip().isdigit():
                continue
            if not token or not values:
                continue
            if wanted is not None and token not in wanted:
                continue
            if token in vocabulary:
                continue
            vocabulary[token] = len(vectors) + 1
            vectors.append(th.tensor([float(value) for value in values.split()]))

    if not vectors:
        raise ValueError(f"{path} covers none of the requested tokens")
    width = vectors[0].numel()
    if any(vector.numel() != width for vector in vectors):
        raise ValueError(f"{path} mixes vectors of different widths")

    missing = (
        [] if wanted is None or pretrained_only else sorted(wanted - set(vocabulary))
    )
    for token in missing:
        vocabulary[token] = len(vectors) + 1
        vectors.append(th.randn(width, generator=generator))

    return vocabulary, th.stack([th.zeros(width), *vectors])
