from pathlib import Path

import lightning as L
import torch as th
from cinnamon.registry import Registry
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import GRAT
from pyhighlights.configurations.spp import GRU_GRAT


def batch() -> InputData:
    return InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )


def test_registered_gru_grat_guidance_and_staged_training():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    data = batch()
    model = Registry.from_key(GRU_GRAT, pretrain_epochs=1, guide_decay=0.2)
    assert isinstance(model, GRAT)
    assert model.automatic_optimization is False
    assert model.selector_backbone is not model.predictor_backbone
    assert model.guide_factor == 1.0
    model._model_steps.add_(1)
    assert model.guide_factor == 1.0
    model._model_steps.add_(1)
    assert model.guide_factor == 0.8
    model._model_steps.zero_()

    output = model(data)
    guider_output = model.guider(data)
    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_logits.shape == (2, 1, 4, 2)
    assert guider_output.attention.shape == (2, 4)
    assert guider_output.class_logits.shape == (2, 2)
    assert th.allclose(guider_output.attention.sum(dim=1), th.ones(2))
    assert not guider_output.attention[~data.mask.bool()].any()

    model_total, losses = model.model_loss(data, output, guider_output)
    assert set(losses) == {"classification", "sparsity", "contiguity", "guide", "jsd"}
    model_total.backward()
    assert any(parameter.grad is not None for parameter in model.selectors.parameters())
    assert all(parameter.grad is None for parameter in model.guider.parameters())

    model.zero_grad(set_to_none=True)
    guider_total, guider_losses = model.guider_loss(data, model.guider(data))
    assert set(guider_losses) == {"classification"}
    guider_total.backward()
    assert any(parameter.grad is not None for parameter in model.guider.parameters())
    assert all(parameter.grad is None for parameter in model.selectors.parameters())

    model = Registry.from_key(GRU_GRAT, pretrain_epochs=1, guide_decay=0.2)
    model_parameters = [
        *model.selector_backbones.parameters(),
        *model.selectors.parameters(),
        *model.predictor_backbone.parameters(),
        *model.predictor.parameters(),
    ]
    model_before = [parameter.detach().clone() for parameter in model_parameters]
    guider_before = [
        parameter.detach().clone() for parameter in model.guider.parameters()
    ]
    trainer = L.Trainer(
        max_epochs=2,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader([data], batch_size=None))

    assert model._model_steps.item() == 1
    assert model.guide_factor == 1.0
    assert any(
        not th.equal(before, after)
        for before, after in zip(model_before, model_parameters)
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(guider_before, model.guider.parameters())
    )
