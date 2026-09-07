import torch as th
from cinnamon.configuration import Configuration
from cinnamon.registry import Registry

from pyhighlights.components.models import InputData, SPPOutput
from pyhighlights.utility.losses import HighlightContiguityLoss, HighlightLoss, JSDiv


def test_highlight_loss_uses_binary_logits_and_ignores_padding():
    Registry.initialize()
    cross_entropy = Registry.register_configuration(
        config=Configuration.default(),
        name="cross_entropy",
        namespace="tests",
        component="torch.nn.CrossEntropyLoss",
    )
    Registry.dag_resolution()

    loss = HighlightLoss(name="highlight", loss=cross_entropy)
    logits = th.tensor([[[3.0, 0.0], [0.0, 3.0], [2.0, 1.0]]], requires_grad=True)
    batch = InputData(
        features=th.zeros((1, 3), dtype=th.long),
        mask=th.tensor([[1.0, 1.0, 0.0]]),
        sample_ids=th.tensor([0]),
        y_true=th.tensor([0]),
        highlight_true=th.tensor([[0, 1, 1]]),
    )
    output = SPPOutput(
        class_logits=th.zeros((1, 2)),
        highlight_logits=logits,
        highlight_mask=th.zeros((1, 3)),
    )

    value = loss(batch, output)
    expected = th.nn.functional.cross_entropy(
        logits[:, :2].reshape(-1, 2), th.tensor([0, 1])
    )

    assert th.allclose(value, expected)
    value.backward()
    assert logits.grad is not None


def test_contiguity_loss_ignores_padding_boundaries():
    loss = HighlightContiguityLoss(name="contiguity")
    batch = InputData(
        features=th.zeros((1, 4), dtype=th.long),
        mask=th.tensor([[1.0, 1.0, 0.0, 0.0]]),
        sample_ids=th.tensor([0]),
        y_true=th.tensor([0]),
        highlight_true=th.full((1, 4), -1),
    )
    output = SPPOutput(
        class_logits=th.zeros((1, 2)),
        highlight_logits=th.zeros((1, 4, 2)),
        highlight_mask=th.tensor([[1.0, 1.0, 0.0, 1.0]]),
    )

    assert loss(batch, output) == 0


def test_js_divergence_is_symmetric():
    divergence = JSDiv()
    p = th.tensor([[2.0, -1.0]])
    q = th.tensor([[-1.0, 2.0]])

    assert th.allclose(divergence(p, p), th.zeros(()), atol=1e-7)
    assert th.allclose(divergence(p, q), divergence(q, p))
    assert divergence(p, q) > 0
