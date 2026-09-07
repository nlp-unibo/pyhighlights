import pytest
import torch as th
from cinnamon.configuration import Configuration
from cinnamon.registry import Registry

from pyhighlights.utility.losses import (
    ContiguityPenalty,
    JSDiv,
    Loss,
    MaskedCrossEntropy,
    compute_losses,
)


def register(component: str, name: str) -> object:
    return Registry.register_configuration(
        config=Configuration.default(),
        name=name,
        namespace="tests",
        component=component,
    )


def test_masked_cross_entropy_ignores_padding_and_unlabelled_tokens():
    logits = th.tensor([[[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]]], requires_grad=True)
    value = MaskedCrossEntropy()(
        logits, th.tensor([[0, 1, 1]]), th.tensor([[1.0, 1.0, 0.0]])
    )
    expected = th.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 2), th.tensor([0, 1])
    )

    assert th.allclose(value, expected)
    value.backward()
    assert logits.grad is not None


def test_contiguity_penalty_ignores_padding_boundaries():
    penalty = ContiguityPenalty()
    selection = th.tensor([[1.0, 1.0, 0.0, 1.0]])
    mask = th.tensor([[1.0, 1.0, 0.0, 0.0]])

    assert penalty(selection, mask) == 0


def test_js_divergence_is_symmetric():
    divergence = JSDiv()
    p = th.tensor([[2.0, -1.0]])
    q = th.tensor([[-1.0, 2.0]])

    assert th.allclose(divergence(p, p), th.zeros(()), atol=1e-7)
    assert th.allclose(divergence(p, q), divergence(q, p))
    assert divergence(p, q) > 0


def test_a_criterion_is_reusable_across_bindings():
    Registry.initialize()
    cross_entropy = register("torch.nn.CrossEntropyLoss", "cross_entropy")
    Registry.dag_resolution()

    selected = Loss(
        name="selected", loss=cross_entropy, inputs=["class_logits", "y_true"]
    )
    full = Loss(
        name="full",
        loss=cross_entropy,
        inputs=["full_class_logits", "y_true"],
        coefficient=2.0,
    )
    values = {
        "class_logits": th.tensor([[2.0, 0.0]]),
        "full_class_logits": th.tensor([[0.0, 2.0]]),
        "y_true": th.tensor([0]),
    }

    total, losses = compute_losses([selected, full], values)

    assert set(losses) == {"selected", "full"}
    assert th.allclose(total, losses["selected"] + 2 * losses["full"])
    assert not th.allclose(losses["selected"], losses["full"])


def test_compute_losses_scales_by_name_and_skips_disabled_losses():
    Registry.initialize()
    sparsity = register("pyhighlights.utility.losses.SparsityPenalty", "sparsity")
    contiguity = register("pyhighlights.utility.losses.ContiguityPenalty", "contiguity")
    Registry.dag_resolution()

    inputs = ["highlight_mask", "mask"]
    penalty = Loss(name="sparsity", loss=sparsity, inputs=inputs, coefficient=3.0)
    disabled = Loss(name="contiguity", loss=contiguity, inputs=inputs, enabled=False)
    values = {
        "highlight_mask": th.tensor([[1.0, 0.0]]),
        "mask": th.tensor([[1.0, 1.0]]),
    }

    total, losses = compute_losses([penalty, disabled], values, scales={"sparsity": 2})

    assert set(losses) == {"sparsity"}
    assert th.allclose(total, losses["sparsity"] * 6)

    empty_total, empty_losses = compute_losses([disabled], values)
    assert empty_losses == {}
    assert empty_total == 0

    with pytest.raises(KeyError, match="mask"):
        penalty({"highlight_mask": th.ones((1, 2))})
