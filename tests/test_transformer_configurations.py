import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.data import HuggingFaceTokenizer
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import (
    FR,
    GRAT,
    MCD,
    MGR,
    GenSPP,
    TransformerBackbone,
)
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import (
    GRU_FR,
    TOY,
    TRANSFORMER_FR,
    TRANSFORMER_GENSPP,
    TRANSFORMER_GRAT,
    TRANSFORMER_MCD,
    TRANSFORMER_MGR,
)


class FakeTransformer(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=6)
        self.embedding = th.nn.Embedding(100, 6)
        self.projection = th.nn.Linear(6, 6)

    def resize_token_embeddings(self, num_features: int):
        self.embedding = th.nn.Embedding(num_features, 6)

    def forward(self, input_ids: th.Tensor, attention_mask: th.Tensor):
        mask = attention_mask.to(self.embedding.weight.dtype).unsqueeze(-1)
        states = self.embedding(input_ids)
        context = (states * mask).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp_min(1)
        return SimpleNamespace(last_hidden_state=self.projection(states + context))


def test_a_task_tokenizes_with_the_model_card_it_names(monkeypatch):
    """Naming a card swaps the fitted vocabulary for that model's tokenizer."""

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, card, **kwargs):
            assert card == "distilbert-base-uncased"
            return SimpleNamespace(name_or_path=card, is_fast=True, pad_token_id=0)

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    task = SPPTask(
        loader=TOY,
        model=GRU_FR,
        pretrained_model_card="distilbert-base-uncased",
    )
    assert isinstance(task.tokenizer({}), HuggingFaceTokenizer)


def test_transformer_backbone_reports_missing_optional_dependency(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(ImportError, match=r"pyhighlights\[transformers\]"):
        TransformerBackbone("unused")


class FakeAutoModel:
    @classmethod
    def from_pretrained(cls, pretrained_model_card: str):
        assert pretrained_model_card == "distilbert-base-uncased"
        return FakeTransformer()


def test_transformer_registrations_are_algorithm_interchangeable(monkeypatch):
    transformers = ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    models = [
        Registry.from_key(TRANSFORMER_FR),
        Registry.from_key(TRANSFORMER_GENSPP),
        Registry.from_key(TRANSFORMER_MGR),
        Registry.from_key(TRANSFORMER_MCD),
        Registry.from_key(TRANSFORMER_GRAT),
    ]
    fr, genspp, mgr, mcd, grat = models
    assert isinstance(fr, FR)
    assert isinstance(genspp, GenSPP)
    assert isinstance(mgr, MGR)
    assert isinstance(mcd, MCD)
    assert isinstance(grat, GRAT)
    assert all(
        isinstance(backbone, TransformerBackbone)
        for model in models
        for backbone in model.selector_backbones
    )
    assert fr.selector_backbone is fr.predictor_backbone
    assert genspp.selector_backbone is not genspp.predictor_backbone
    assert not any(
        parameter.requires_grad
        for parameter in genspp.selector_backbone.transformer.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in genspp.predictor_backbone.transformer.parameters()
    )
    assert len(mgr.selector_backbones) == 3
    assert mcd.selector_backbone is not mcd.predictor_backbone
    assert grat.selector_backbone is not grat.predictor_backbone
    assert isinstance(grat.guider.backbone, TransformerBackbone)

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )
    expected_heads = (1, 1, 3, 1, 1)
    for model, heads in zip(models, expected_heads):
        output = model(batch)
        assert output.class_logits.shape == (2, heads, 2)
        assert output.highlight_logits.shape == (2, heads, 4, 2)
        assert not output.highlight_mask[:, :, 3].any()

    th.manual_seed(0)
    classification_loss = fr.losses[0](fr.head_namespace(batch, fr(batch)))
    classification_loss.backward()
    selector_gradient = fr.selectors[0].selector[-1].weight.grad
    assert selector_gradient is not None
    assert selector_gradient.abs().sum() > 0
    assert fr.selector_backbone.transformer.embedding.weight.grad is not None
    assert fr.predictor.predictor[-1].weight.grad is not None
