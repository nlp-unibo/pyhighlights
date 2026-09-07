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
from pyhighlights.configurations.spp import GRU_FR


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


def test_collator_aligns_subwords_missing_annotations_and_padding():
    collator = HighlightCollator(SubwordTokenizer(), max_length=3)
    batch = collator(
        [
            HighlightExample(7, ["great", "stay"], 1, [1, 0]),
            HighlightExample(8, ["bad"], 0),
        ]
    )

    assert th.equal(batch.features, th.tensor([[10, 11, 12], [12, 0, 0]]))
    assert th.equal(batch.mask, th.tensor([[1, 1, 1], [1, 0, 0]]))
    assert th.equal(batch.highlight_true, th.tensor([[1, 1, 0], [-1, -1, -1]]))
    assert th.equal(batch.sample_ids, th.tensor([7, 8]))
    assert th.equal(batch.y_true, th.tensor([1, 0]))

    empty = HighlightCollator(VocabularyTokenizer({}))([HighlightExample(9, [], 0, [])])
    assert empty.features.shape == empty.mask.shape == (1, 1)
    assert empty.mask.sum() == 0
    assert empty.highlight_true.item() == -1


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
                "add_special_tokens": False,
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
