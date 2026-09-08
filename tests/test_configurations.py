from pathlib import Path

import lightning as L
import pytest
import torch as th
from cinnamon.registry import Registry
from torch.utils.data import DataLoader

import pyhighlights
from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import FR, MCD, MGR
from pyhighlights.configurations.keys import GRU_FR, GRU_MCD, GRU_MGR


def test_registered_gru_fr_forward_backward_and_optimizer():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_FR)
    assert isinstance(model, FR)

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )

    model.train()
    output = model(batch)
    loss, losses = model.compute_loss(batch, output)
    loss.backward()

    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_logits.shape == (2, 1, 4, 2)
    assert output.highlight_mask.shape == (2, 1, 4)
    assert set(losses) == {"classification", "sparsity", "contiguity"}
    assert model.selector_backbone.embedding.weight.grad is not None
    assert model.selectors[0].selector[0].weight.grad is not None
    predictor_weight = model.predictor.predictor[-1].weight
    assert predictor_weight.grad is not None
    assert predictor_weight.grad.abs().sum() > 0

    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, th.optim.Adam)
    previous_weight = predictor_weight.detach().clone()
    optimizer.step()
    assert not th.equal(previous_weight, predictor_weight)

    model.eval()
    with th.no_grad():
        first = model(batch).highlight_mask
        second = model(batch).highlight_mask
        short_states = model.selector_backbone.encode(
            th.tensor([[1, 2, 3]]), th.ones((1, 3))
        )
        padded_states = model.selector_backbone.encode(
            th.tensor([[1, 2, 3, 0, 0]]),
            th.tensor([[1.0, 1.0, 1.0, 0.0, 0.0]]),
        )
    assert th.equal(first, second)
    assert th.allclose(short_states, padded_states[:, :3], atol=1e-6)


def test_registered_gru_mcd_alternates_generator_and_predictor_updates():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_MCD)
    assert isinstance(model, MCD)
    assert model.automatic_optimization is False
    assert model.selector_backbone is not model.predictor_backbone

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )
    output = model(batch)
    assert output.class_logits.shape == (2, 1, 2)
    assert output.highlight_logits.shape == (2, 1, 4, 2)

    for loss in model.rationale_losses:
        loss.enabled = False
    classifier_total, classifier_losses, _ = model.classifier_phase_loss(batch)
    assert set(classifier_losses) == {"classification", "full_classification"}
    classifier_total.backward()
    assert all(
        parameter.grad is None for parameter in model.selector_backbone.parameters()
    )
    assert any(parameter.grad is not None for parameter in model.predictor.parameters())

    model.zero_grad(set_to_none=True)
    for loss in model.rationale_losses:
        loss.enabled = True
    generator_total, generator_losses, _ = model.generator_phase_loss(batch)
    assert set(generator_losses) == {"sparsity", "contiguity", "discrepancy"}
    generator_total.backward()
    assert any(
        parameter.grad is not None for parameter in model.selector_backbone.parameters()
    )
    assert all(
        parameter.grad is None for parameter in model.predictor_backbone.parameters()
    )
    assert all(parameter.grad is None for parameter in model.predictor.parameters())

    model = Registry.from_key(GRU_MCD)
    generator_parameters = [
        *model.selector_backbones.parameters(),
        *model.selectors.parameters(),
    ]
    predictor_parameters = [
        *model.predictor_backbone.parameters(),
        *model.predictor.parameters(),
    ]
    generator_before = [
        parameter.detach().clone() for parameter in generator_parameters
    ]
    predictor_before = [
        parameter.detach().clone() for parameter in predictor_parameters
    ]
    trainer = L.Trainer(
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_dataloaders=DataLoader([batch], batch_size=None))
    assert any(
        not th.equal(before, after)
        for before, after in zip(generator_before, generator_parameters)
    )
    assert any(
        not th.equal(before, after)
        for before, after in zip(predictor_before, predictor_parameters)
    )


def test_registered_gru_mgr_has_independent_generators_and_head_policy():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    model = Registry.from_key(GRU_MGR)
    assert isinstance(model, MGR)
    assert len(model.selectors) == len(model.selector_backbones) == 3
    assert len({id(backbone) for backbone in model.selector_backbones}) == 3
    assert len({id(selector) for selector in model.selectors}) == 3
    assert all(
        model.predictor_backbone is not backbone
        for backbone in model.selector_backbones
    )

    optimizer = model.configure_optimizers()
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx(
        [1e-3 / 3, 1e-3, 2e-3, 3e-3]
    )
    assert {id(parameter) for parameter in optimizer.param_groups[0]["params"]} == {
        id(parameter)
        for parameter in [
            *model.predictor_backbone.parameters(),
            *model.predictor.parameters(),
        ]
    }
    for index, (backbone, selector) in enumerate(
        zip(model.selector_backbones, model.selectors), start=1
    ):
        assert {
            id(parameter) for parameter in optimizer.param_groups[index]["params"]
        } == {
            id(parameter)
            for parameter in [*backbone.parameters(), *selector.parameters()]
        }

    batch = InputData(
        features=th.tensor([[1, 2, 3, 0], [4, 5, 0, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 4), -1),
    )

    model.train()
    output = model(batch)
    summed_loss, summed_parts = model.compute_loss(batch, output)
    model.loss_reduction = "mean"
    mean_loss, mean_parts = model.compute_loss(batch, output)

    assert output.class_logits.shape == (2, 3, 2)
    assert output.highlight_logits.shape == (2, 3, 4, 2)
    assert output.highlight_mask.shape == (2, 3, 4)
    assert th.allclose(mean_loss * 3, summed_loss)
    assert all(
        th.allclose(mean_parts[name] * 3, summed_parts[name]) for name in summed_parts
    )

    mean_loss.backward()
    assert all(
        backbone.embedding.weight.grad is not None
        for backbone in model.selector_backbones
    )
    assert model.predictor_backbone.embedding.weight.grad is not None

    model.inference_head = 1
    model.eval()
    with th.no_grad():
        expected = model.forward_one_head(batch, selector_idx=1)
        actual = model.validation_forward(batch)
    assert actual.class_logits.shape == (2, 1, 2)
    assert th.equal(actual.class_logits, expected.class_logits)
    assert th.equal(actual.highlight_mask, expected.highlight_mask)
