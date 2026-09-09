from pathlib import Path

import pytest
import torch as th
from cinnamon.configuration import Configuration
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.configurations.keys import CROSS_ENTROPY
from pyhighlights.utility.losses import (
    ContiguityPenalty,
    CrossEntropy,
    JSDiv,
    Loss,
    MaskedCrossEntropy,
    SparsityPenalty,
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


def test_supervision_appends_the_highlight_loss_and_guides_the_first_head_only():
    """One annotation guides one head: the one every metric scores."""
    import pyhighlights
    from pyhighlights.components.models import InputData
    from pyhighlights.configurations.keys import GRU_MGR, HIGHLIGHT_LOSS

    Registry.build(directory=Path(pyhighlights.__file__).parent)
    plain = Registry.from_key(GRU_MGR)
    guided = Registry.from_key(
        GRU_MGR,
        supervise_highlights=True,
        highlight_loss=HIGHLIGHT_LOSS,
        highlight_coefficient=0.5,
    )
    assert [loss.name for loss in plain.losses] == [
        "classification",
        "sparsity",
        "contiguity",
    ]
    assert guided.losses[guided.supervised].name == "highlight"
    assert guided.losses[guided.supervised].coefficient == 0.5
    assert plain.supervised is None

    batch = InputData(
        features=th.tensor([[4, 5, 6, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0, 0.0]]),
        sample_ids=th.tensor([0]),
        y_true=th.tensor([1]),
        highlight_true=th.tensor([[0, 1, 1, -1]]),
    )
    guided.eval()
    with th.no_grad():
        output = guided(batch)
        _, terms = guided.compute_loss(batch, output)

        criterion = MaskedCrossEntropy()
        heads = list(output.unbind(dim=1))
        first = criterion(heads[0].highlight_logits, batch.highlight_true, batch.mask)
        every = sum(
            criterion(head.highlight_logits, batch.highlight_true, batch.mask)
            for head in heads
        )

    assert len(heads) == 3
    # The other terms are summed over every head; the supervised one is not.
    assert terms["highlight"] == pytest.approx(first.item())
    assert terms["highlight"] != pytest.approx(every.item())

    with th.no_grad():
        sparsity = sum(
            SparsityPenalty()(head.highlight_mask, batch.mask) for head in heads
        )
    assert terms["sparsity"] == pytest.approx(sparsity.item())


def test_unweighted_cross_entropy_is_the_one_torch_ships():
    """Nothing about the default changes by wrapping it."""
    logits = th.tensor([[2.0, -1.0], [0.5, 0.5]])
    targets = th.tensor([0, 1])

    assert CrossEntropy()(logits, targets) == pytest.approx(
        float(th.nn.CrossEntropyLoss()(logits, targets))
    )


def test_class_weights_make_the_rare_class_dominate_the_batch():
    """The point of weighting: a corpus that is 1% positive.

    Weighted ``mean`` reduction divides by the sum of the weights in the
    batch, not by its size, so a weight changes how much a class counts
    *relative to the others* and a single-class batch is unaffected by it.
    """
    logits = th.tensor([[2.0, -1.0], [2.0, -1.0]])
    plain = CrossEntropy()
    weighted = CrossEntropy(weight=[1.0, 20.0])

    on_common = float(plain(logits[:1], th.tensor([0])))
    on_rare = float(plain(logits[:1], th.tensor([1])))
    batch = th.tensor([0, 1])

    assert plain(logits, batch) == pytest.approx((on_common + on_rare) / 2)
    assert weighted(logits, batch) == pytest.approx(
        (on_common + 20 * on_rare) / 21, rel=1e-5
    )
    # Which is to say the rare class now carries almost all of the loss.
    assert weighted(logits, batch) > plain(logits, batch)


def test_the_weights_follow_the_model_and_stay_out_of_its_checkpoint():
    """A weight tensor left behind is a crash at the first batch."""
    criterion = CrossEntropy(weight=[1.0, 20.0])
    module = th.nn.Sequential(criterion)

    assert criterion.weight.dtype == th.get_default_dtype()
    # A buffer moves with the module; a plain attribute would not.
    assert dict(module.named_buffers())["0.weight"] is criterion.weight
    # Configured rather than trained, so a checkpoint does not carry it and a
    # run configured without weights can load one saved with them.
    assert module.state_dict() == {}


def test_the_registered_criterion_takes_its_weights_from_the_configuration():
    Registry.build(directory=Path(pyhighlights.__file__).parent)

    criterion = Registry.from_key(CROSS_ENTROPY)
    assert isinstance(criterion, CrossEntropy)
    assert criterion.weight is None

    weighted = Registry.from_key(CROSS_ENTROPY, weight=[1.0, 20.0])
    assert weighted.weight.tolist() == [1.0, 20.0]
