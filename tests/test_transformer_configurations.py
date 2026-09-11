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
    MRD,
    GenSPP,
    StackedBackbone,
    TransformerBackbone,
)
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import (
    FROZEN_TRANSFORMER_BACKBONE,
    GRU_FR,
    STACKED_BACKBONE,
    TOY,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_FR,
    TRANSFORMER_GENSPP,
    TRANSFORMER_GRAT,
    TRANSFORMER_MCD,
    TRANSFORMER_MGR,
    TRANSFORMER_MRD,
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
        Registry.from_key(TRANSFORMER_MRD),
        Registry.from_key(TRANSFORMER_GRAT),
    ]
    fr, genspp, mgr, mcd, mrd, grat = models
    assert isinstance(fr, FR)
    assert isinstance(genspp, GenSPP)
    assert isinstance(mgr, MGR)
    assert isinstance(mcd, MCD)
    assert isinstance(mrd, MRD)
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
    assert mrd.selector_backbone is not mrd.predictor_backbone
    assert grat.selector_backbone is not grat.predictor_backbone
    assert isinstance(grat.guider.backbone, TransformerBackbone)

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )
    expected_heads = (1, 1, 3, 1, 1, 1)
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


def test_a_frozen_transformer_backbone_is_a_key_of_its_own(monkeypatch):
    """Trainability is addressable, so a study need not re-register the encoder.

    ``freeze_transformer`` was always a parameter, but a parameter behind a key
    cannot be reached from outside: build arguments reach the component a
    caller builds, not the ones built underneath it.
    """
    transformers = ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    frozen = Registry.from_key(FROZEN_TRANSFORMER_BACKBONE)
    assert isinstance(frozen, TransformerBackbone)
    assert not any(
        parameter.requires_grad for parameter in frozen.transformer.parameters()
    )

    # The default is unchanged: fine-tuning is what a transformer arm is for.
    trainable = Registry.from_key(TRANSFORMER_BACKBONE)
    assert all(
        parameter.requires_grad for parameter in trainable.transformer.parameters()
    )

    # Frozen or not, it is the same encoder: an algorithm reads it through the
    # backbone contract and cannot tell.
    data = InputData(
        features=th.tensor([[1, 2, 3]]),
        mask=th.tensor([[1.0, 1.0, 1.0]]),
        sample_ids=th.arange(1),
        y_true=th.tensor([0]),
        highlight_true=th.full((1, 3), -1),
    )
    states = frozen.encode(data.features, data.mask)
    assert states.shape == (1, 3, frozen.output_size)
    assert not states.requires_grad


def test_a_stacked_backbone_trains_a_gru_over_a_frozen_transformer(monkeypatch):
    """The architecture the papers use, with a transformer where GloVe was.

    Every released select-then-predict implementation encodes with a
    bidirectional GRU over a frozen embedding table, so nothing pretrained is
    fine-tuned and everything trained starts from scratch at one learning
    rate. This keeps that shape.
    """
    transformers = ModuleType("transformers")
    transformers.AutoModel = FakeAutoModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    backbone = Registry.from_key(STACKED_BACKBONE, hidden_size=4)
    assert isinstance(backbone, StackedBackbone)
    # Bidirectional by default, so the states are twice the hidden size --
    # and nothing downstream reads the transformer's width.
    assert backbone.output_size == 8

    # Frozen underneath, trainable on top: one learning rate is correct for
    # every parameter that has a gradient.
    assert not any(
        parameter.requires_grad
        for parameter in backbone.transformer.transformer.parameters()
    )
    assert all(parameter.requires_grad for parameter in backbone.encoder.parameters())

    features = th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]])
    mask = th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
    states = backbone.encode(features, mask)
    assert states.shape == (2, 4, 8)
    # Padding carries nothing, and a gradient reaches the GRU.
    assert not states[~mask.bool()].any()
    states.sum().backward()
    assert any(
        parameter.grad is not None for parameter in backbone.encoder.parameters()
    )

    pooled = backbone.pool(states.detach(), mask)
    assert pooled.shape == (2, 8)

    # A selection reaches the transformer rather than the GRU's input: the
    # predictor has to read the highlight and nothing else.
    selected = backbone.encode(features, mask, selection_mask=mask * 0)
    assert not th.equal(selected, states.detach())
