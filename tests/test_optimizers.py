"""Two learning rates: the encoders, and everything initialized from scratch.

Left unset, a model trains as every published implementation of these
architectures does -- one optimizer at one rate over the whole thing, because
they encode with a GRU over a frozen table and nothing pretrained is ever
fine-tuned. ``encoder_lr`` is for the case those papers never had: a
transformer being fine-tuned underneath a selector that started from random
weights, where one rate is wrong for one of the two.
"""

from pathlib import Path

import pytest
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.configurations.keys import (
    GRU_FR,
    GRU_GRAT,
    GRU_MCD,
    GRU_MGR,
    GRU_MRD,
)

BASE_LR = 1e-3
ENCODER_LR = 2e-5


@pytest.fixture(scope="module", autouse=True)
def registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def rates(optimizer):
    return [group["lr"] for group in optimizer.param_groups]


def test_one_rate_by_default_over_the_whole_model():
    """The published setting: nothing pretrained, so nothing needs sparing."""
    model = Registry.from_key(GRU_FR)
    optimizer = model.configure_optimizers()

    assert rates(optimizer) == [BASE_LR]
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {id(parameter) for parameter in model.parameters()}


def test_the_encoder_rate_applies_to_backbones_and_nothing_else():
    model = Registry.from_key(GRU_FR, encoder_lr=ENCODER_LR)
    optimizer = model.configure_optimizers()

    assert sorted(rates(optimizer)) == [ENCODER_LR, BASE_LR]
    encoders = model.encoder_ids()
    for group in optimizer.param_groups:
        inside = {id(parameter) for parameter in group["params"]} <= encoders
        assert group["lr"] == (ENCODER_LR if inside else BASE_LR)

    # Every parameter still trains, in exactly one group.
    ids = [
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    assert sorted(ids) == sorted(id(parameter) for parameter in model.parameters())
    assert len(ids) == len(set(ids))


def test_mgr_keeps_its_generator_scaling_under_two_rates():
    """MGR's rates differ by design; the encoder rate is scaled the same way.

    The predictor trains slower than the generators by the number of them and
    each generator faster than the last, which is the paper's schedule. It is
    a scale on a group, so splitting a group in two has to leave both halves
    on the same scale.
    """
    model = Registry.from_key(GRU_MGR)
    scales = [1 / 3, 1, 2, 3]
    assert rates(model.configure_optimizers()) == [BASE_LR * s for s in scales]

    model = Registry.from_key(GRU_MGR, encoder_lr=ENCODER_LR)
    optimizer = model.configure_optimizers()
    # Four groups become eight: each of the paper's groups holds a backbone
    # and a head.
    assert len(optimizer.param_groups) == 8
    encoders = model.encoder_ids()
    found = []
    for group in optimizer.param_groups:
        inside = {id(parameter) for parameter in group["params"]} <= encoders
        found.append((inside, group["lr"]))
    for scale in scales:
        assert (True, pytest.approx(ENCODER_LR * scale)) in [
            (inside, pytest.approx(lr)) for inside, lr in found
        ]
        assert (False, pytest.approx(BASE_LR * scale)) in [
            (inside, pytest.approx(lr)) for inside, lr in found
        ]


def test_grat_spares_the_guiders_encoder_too():
    """G-RAT holds three encoders, and the guider's is one of them."""
    model = Registry.from_key(GRU_GRAT, encoder_lr=ENCODER_LR)
    guider_optimizer, model_optimizer = model.configure_optimizers()

    assert sorted(rates(guider_optimizer)) == [ENCODER_LR, BASE_LR]
    assert sorted(rates(model_optimizer)) == [ENCODER_LR, BASE_LR]

    guider_encoder = {id(parameter) for parameter in model.guider.backbone.parameters()}
    assert guider_encoder <= model.encoder_ids()
    for group in guider_optimizer.param_groups:
        inside = {id(parameter) for parameter in group["params"]} <= guider_encoder
        assert group["lr"] == (ENCODER_LR if inside else BASE_LR)


def test_a_rate_that_cannot_train_anything_is_refused():
    with pytest.raises(ValueError, match="encoder_lr"):
        Registry.from_key(GRU_FR, encoder_lr=0.0)


@pytest.mark.parametrize("key", [GRU_FR, GRU_MCD, GRU_MGR, GRU_MRD, GRU_GRAT])
def test_every_architecture_accepts_an_encoder_rate(key):
    """MCD and G-RAT declare the shared field set instead of inheriting it.

    So a parameter added to ``SPPModelConfig`` reaches FR and GenSPP and
    misses those two, and asking for it raises rather than being ignored --
    which is the trap this test exists to catch next time.
    """
    model = Registry.from_key(key, encoder_lr=ENCODER_LR)
    optimizer = model.configure_optimizers()
    optimizers = optimizer if isinstance(optimizer, list) else [optimizer]

    for one in optimizers:
        encoders = model.encoder_ids()
        rates = {
            ({id(p) for p in group["params"]} <= encoders): group["lr"]
            for group in one.param_groups
        }
        # Both halves are present and the encoder half is the slower one.
        assert set(rates) == {True, False}
        assert rates[True] < rates[False]
