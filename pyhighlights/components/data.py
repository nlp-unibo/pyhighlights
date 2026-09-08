from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Iterable, Mapping, Protocol, Sequence

import torch as th
from torch.utils.data import Dataset

from pyhighlights.components.models import InputData

#: Columns every corpus frame carries, in order. A loader parses into them and
#: a preprocessor returns them, so a split is the same shape wherever it came
#: from. ``highlights`` is ``None`` where a split has no annotation, and
#: ``label`` is ``None`` only until a preprocessor has resolved one.
COLUMNS = ("sample_id", "text", "tokens", "label", "highlights")


@dataclass(frozen=True)
class HighlightExample:
    sample_id: int
    tokens: Sequence[str]
    label: int
    highlights: Sequence[int] | None = None

    def __post_init__(self):
        tokens = tuple(self.tokens)
        highlights = None if self.highlights is None else tuple(self.highlights)
        if not isinstance(self.sample_id, Integral) or not isinstance(
            self.label, Integral
        ):
            raise TypeError("sample_id and label must be integers")
        if any(not isinstance(token, str) for token in tokens):
            raise TypeError("tokens must contain strings")
        if highlights is not None:
            if len(highlights) != len(tokens):
                raise ValueError("highlights must align with tokens")
            if any(value not in (0, 1) for value in highlights):
                raise ValueError("highlights must contain only 0 or 1")
        object.__setattr__(self, "tokens", tokens)
        object.__setattr__(self, "highlights", highlights)


@dataclass(frozen=True)
class TokenizedExample:
    input_ids: Sequence[int]
    word_ids: Sequence[int | None]

    def __post_init__(self):
        input_ids = tuple(self.input_ids)
        word_ids = tuple(self.word_ids)
        if len(input_ids) != len(word_ids):
            raise ValueError("input_ids and word_ids must have equal length")
        if any(not isinstance(value, Integral) or value < 0 for value in input_ids):
            raise ValueError("input_ids must contain non-negative integers")
        if any(
            value is not None and (not isinstance(value, Integral) or value < 0)
            for value in word_ids
        ):
            raise ValueError("word_ids must contain non-negative integers or None")
        object.__setattr__(self, "input_ids", input_ids)
        object.__setattr__(self, "word_ids", word_ids)


class HighlightTokenizer(Protocol):
    pad_token_id: int

    def encode(
        self, tokens: Sequence[str], max_length: int | None = None
    ) -> TokenizedExample: ...


class VocabularyTokenizer:
    """One-token-to-one-id encoder for GRU backbones."""

    def __init__(
        self,
        vocabulary: Mapping[str, int],
        unknown_token_id: int = 0,
        pad_token_id: int = 0,
    ):
        if min([unknown_token_id, pad_token_id, *vocabulary.values()]) < 0:
            raise ValueError("token ids must be non-negative")
        self.vocabulary = dict(vocabulary)
        self.unknown_token_id = unknown_token_id
        self.pad_token_id = pad_token_id

    def encode(
        self, tokens: Sequence[str], max_length: int | None = None
    ) -> TokenizedExample:
        tokens = tokens if max_length is None else tokens[:max_length]
        return TokenizedExample(
            input_ids=[
                self.vocabulary.get(token, self.unknown_token_id) for token in tokens
            ],
            word_ids=list(range(len(tokens))),
        )


class HuggingFaceTokenizer:
    """Fast-tokenizer adapter preserving source-token alignment."""

    def __init__(self, pretrained_model_card: str, **tokenizer_kwargs):
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "HuggingFaceTokenizer requires pyhighlights[transformers]"
            ) from error

        if tokenizer_kwargs.pop("use_fast", True) is not True:
            raise ValueError("HuggingFaceTokenizer requires a fast tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_card, use_fast=True, **tokenizer_kwargs
        )
        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("HuggingFaceTokenizer requires a fast tokenizer")
        if self.tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must define pad_token_id")
        self.pad_token_id = self.tokenizer.pad_token_id

    def encode(
        self, tokens: Sequence[str], max_length: int | None = None
    ) -> TokenizedExample:
        kwargs = {
            "is_split_into_words": True,
            "add_special_tokens": False,
            "return_attention_mask": False,
        }
        if max_length is not None:
            kwargs.update(truncation=True, max_length=max_length)
        encoded = self.tokenizer(list(tokens), **kwargs)
        try:
            word_ids = encoded.word_ids()
        except (AttributeError, ValueError) as error:
            raise ValueError(
                "tokenizer must provide word_ids; use a fast tokenizer"
            ) from error
        return TokenizedExample(
            input_ids=encoded["input_ids"],
            word_ids=word_ids,
        )


class HighlightDataset(Dataset):
    def __init__(self, examples: Iterable[HighlightExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> HighlightExample:
        return self.examples[index]


class HighlightCollator:
    """Tokenize, align highlights to subtokens, and dynamically pad a batch."""

    def __init__(
        self,
        tokenizer: HighlightTokenizer,
        max_length: int | None = None,
    ):
        if max_length is not None and max_length < 1:
            raise ValueError("max_length must be positive")
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, examples: Sequence[HighlightExample]) -> InputData:
        if not examples:
            raise ValueError("cannot collate an empty batch")
        encoded = [
            self.tokenizer.encode(example.tokens, self.max_length)
            for example in examples
        ]
        if self.max_length is not None:
            encoded = [
                TokenizedExample(
                    item.input_ids[: self.max_length],
                    item.word_ids[: self.max_length],
                )
                for item in encoded
            ]
        width = max(max(len(item.input_ids) for item in encoded), 1)

        features = []
        masks = []
        highlights = []
        for example, item in zip(examples, encoded):
            input_ids = list(item.input_ids[:width])
            word_ids = list(item.word_ids[:width])
            for word_id in word_ids:
                if word_id is not None and not 0 <= word_id < len(example.tokens):
                    raise ValueError("word_id is outside source-token range")

            padding = width - len(input_ids)
            features.append(input_ids + [self.tokenizer.pad_token_id] * padding)
            masks.append(
                [word_id is not None for word_id in word_ids] + [False] * padding
            )
            highlights.append(
                [
                    -1
                    if word_id is None or example.highlights is None
                    else example.highlights[word_id]
                    for word_id in word_ids
                ]
                + [-1] * padding
            )

        return InputData(
            features=th.tensor(features, dtype=th.long),
            mask=th.tensor(masks, dtype=th.float32),
            sample_ids=th.tensor([example.sample_id for example in examples]),
            y_true=th.tensor([example.label for example in examples]),
            highlight_true=th.tensor(highlights, dtype=th.long),
        )
