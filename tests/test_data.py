import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch as th
from cinnamon.registry import Registry
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components import (
    HighlightCollator,
    HighlightDataset,
    HighlightExample,
    HuggingFaceTokenizer,
    TokenizedExample,
    VocabularyTokenizer,
)
from pyhighlights.configurations.keys import GRU_FR


class SubwordTokenizer:
    pad_token_id = 0

    def encode(self, tokens, max_length=None):
        ids = [
            piece
            for token in tokens
            for piece in ([10, 11] if token == "great" else [12])
        ]
        word_ids = [
            index
            for index, token in enumerate(tokens)
            for _ in range(2 if token == "great" else 1)
        ]
        if max_length is not None:
            ids = ids[:max_length]
            word_ids = word_ids[:max_length]
        return TokenizedExample(ids, word_ids)


class SpecialTokenTokenizer:
    """One subtoken per word, bracketed by a special token at each end."""

    pad_token_id = 0

    def encode(self, tokens, max_length=None):
        tokens = tokens if max_length is None else tokens[:max_length]
        return TokenizedExample(
            input_ids=[101, *range(10, 10 + len(tokens)), 102],
            word_ids=[None, *range(len(tokens)), None],
        )


def test_collator_aligns_subwords_missing_annotations_and_padding():
    collator = HighlightCollator(SubwordTokenizer(), max_length=3)
    batch = collator(
        [
            HighlightExample(7, ["great", "stay"], 1, [1, 0]),
            HighlightExample(8, ["bad"], 0),
        ]
    )

    # Subtoken axis: what the encoder reads. "great" splits in two, so row 0
    # is three positions wide and row 1 is padded to match.
    assert th.equal(batch.features, th.tensor([[10, 11, 12], [12, 0, 0]]))
    assert th.equal(batch.attention_mask, th.tensor([[1, 1, 1], [1, 0, 0]]))
    assert th.equal(batch.word_ids, th.tensor([[0, 0, 1], [0, -1, -1]]))

    # Word axis: what a selection is made over. Two words and one, so the
    # annotation is the corpus's own rather than one spread over subtokens.
    assert th.equal(batch.mask, th.tensor([[1, 1], [1, 0]]))
    assert th.equal(batch.highlight_true, th.tensor([[1, 0], [-1, -1]]))
    assert th.equal(batch.sample_ids, th.tensor([7, 8]))
    assert th.equal(batch.y_true, th.tensor([1, 0]))

    empty = HighlightCollator(VocabularyTokenizer({}))([HighlightExample(9, [], 0, [])])
    assert empty.features.shape == empty.mask.shape == (1, 1)
    assert empty.mask.sum() == 0
    assert empty.highlight_true.item() == -1


def test_the_word_axis_stops_where_truncation_did():
    """A word past the cut was never encoded, so it cannot be selected.

    The word axis is as wide as the longest surviving word count, not as the
    longest document: a mask over words the encoder never saw would invite a
    selection nothing could act on, and a sparsity rate measured against a
    denominator half of which does not exist.
    """
    collator = HighlightCollator(SubwordTokenizer(), max_length=2)
    batch = collator([HighlightExample(1, ["great", "stay"], 1, [1, 0])])

    # "great" takes both surviving positions, so only that word is on the axis.
    assert th.equal(batch.word_ids, th.tensor([[0, 0]]))
    assert th.equal(batch.mask, th.tensor([[1.0]]))
    assert th.equal(batch.highlight_true, th.tensor([[1]]))


def test_vocabulary_dataset_batch_runs_registered_model():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    dataset = HighlightDataset(
        [
            HighlightExample(0, ["good", "stay"], 1, [1, 0]),
            HighlightExample(1, ["bad"], 0),
        ]
    )
    collator = HighlightCollator(VocabularyTokenizer({"good": 1, "stay": 2, "bad": 3}))
    batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collator)))
    model = Registry.from_key(GRU_FR)

    output = model(batch)
    loss, _ = model.compute_loss(batch, output)
    loss.backward()

    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_mask.shape == (2, 1, 2)
    assert model.selector_backbone.embedding.weight.grad is not None


def test_huggingface_adapter_preserves_word_ids(monkeypatch):
    class Encoding(dict):
        def word_ids(self):
            return [0, 0, 1]

    class FakeTokenizer:
        is_fast = True
        pad_token_id = 9

        def __call__(self, tokens, **kwargs):
            assert tokens == ["great", "stay"]
            assert kwargs == {
                "is_split_into_words": True,
                "add_special_tokens": True,
                "return_attention_mask": False,
                "truncation": True,
                "max_length": 3,
            }
            return Encoding(input_ids=[4, 5, 6])

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_card, **kwargs):
            assert model_card == "fake"
            assert kwargs == {"use_fast": True, "add_prefix_space": True}
            return FakeTokenizer()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    tokenizer = HuggingFaceTokenizer("fake", add_prefix_space=True)
    assert tokenizer.pad_token_id == 9
    assert tokenizer.encode(["great", "stay"], 3) == TokenizedExample(
        [4, 5, 6], [0, 0, 1]
    )


def test_data_validation_and_optional_dependency_error(monkeypatch):
    input_ids = [1]
    tokenized = TokenizedExample(input_ids, [0])
    input_ids[0] = -1
    assert tokenized.input_ids == (1,)

    with pytest.raises(ValueError, match="align"):
        HighlightExample(0, ["two", "tokens"], 0, [1])
    with pytest.raises(ValueError, match="only 0 or 1"):
        HighlightExample(0, ["token"], 0, [2])

    class SlowAutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return type("SlowTokenizer", (), {"is_fast": False, "pad_token_id": 0})()

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = SlowAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    with pytest.raises(ValueError, match="fast tokenizer"):
        HuggingFaceTokenizer("slow")

    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r"pyhighlights\[transformers\]"):
        HuggingFaceTokenizer("unused")


def test_a_word_is_selected_whole_or_not_at_all():
    """The property the word axis exists for.

    Selecting over subtokens lets a model keep ``un`` and drop ``##fair``,
    which the export then reports as the word ``unfair`` -- a highlight that
    is not what the predictor read. Over words the two cannot disagree: the
    mask the predictor is given is the selection, spread over every piece of
    each word it kept.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_FR)
    collator = HighlightCollator(SubwordTokenizer())
    batch = collator(
        [
            HighlightExample(0, ["great", "stay", "great"], 1, [1, 0, 1]),
            HighlightExample(1, ["stay"], 0, [0]),
        ]
    )

    # Word axis for the selection, subtoken axis for what the encoder reads.
    output = model(batch)
    assert output.highlight_mask.shape[-1] == batch.mask.shape[1] == 3
    assert batch.attention_mask.shape == batch.features.shape

    selection = output.highlight_mask[:, 0]
    spread = model.to_subtokens(selection * batch.mask, batch)
    for row, ids in zip(spread, batch.word_ids):
        for word in ids.unique():
            if word < 0:
                continue
            pieces = row[ids == word]
            assert pieces.min() == pieces.max(), "a word was selected in part"


def test_special_tokens_are_never_selected_but_are_always_read():
    """``[CLS]`` is not a word, so it is not a choice -- and never dropped.

    Removing it from the input was how the library used to keep it
    unselectable, at the cost of running a pretrained encoder off the
    distribution it was trained on.
    """
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_FR)
    collator = HighlightCollator(SpecialTokenTokenizer())
    batch = collator([HighlightExample(0, ["great", "stay"], 1, [1, 0])])

    # The specials bracket the words: attended, but on no word slot.
    assert th.equal(batch.word_ids, th.tensor([[-1, 0, 1, -1]]))
    assert th.equal(batch.attention_mask, th.tensor([[1.0, 1.0, 1.0, 1.0]]))
    assert th.equal(batch.mask, th.tensor([[1.0, 1.0]]))

    # Whatever the selector decides, both specials survive into the predictor.
    for selection in (th.tensor([[0.0, 0.0]]), th.tensor([[1.0, 0.0]])):
        spread = model.to_subtokens(selection, batch)
        assert spread[0, 0] == 1.0 and spread[0, -1] == 1.0
