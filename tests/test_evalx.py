"""EVAL-X: the random selector, the evaluator, and its readings of selections."""

import math
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.codes import SelectionCodeAnalyzer, evaluator_readings
from pyhighlights.components.models.spp.evalx import RandomSelector
from pyhighlights.configurations.keys import (
    GRU_EVALX,
    MLP_SELECTOR,
    TOY_EVALX_TASK,
    TOY_TASK,
)


def valid_rows(lengths, width=12):
    return (
        th.arange(width).expand(len(lengths), width) < th.tensor(lengths)[:, None]
    ).float()


def test_tokens_keep_only_valid_positions_and_at_least_one():
    th.manual_seed(0)
    valid = valid_rows([12, 5, 1, 0] * 50)
    keep = RandomSelector(mode="tokens", min_keep=0.0, max_keep=0.05).sample(valid)
    assert ((keep == 1) <= (valid == 1)).all()
    assert (keep.sum(-1)[valid.sum(-1) > 0] >= 1).all()
    assert (keep.sum(-1)[valid.sum(-1) == 0] == 0).all()


def test_the_keep_rate_is_drawn_from_its_interval():
    th.manual_seed(0)
    valid = th.ones(4000, 20)
    keep = RandomSelector(mode="tokens", min_keep=0.5, max_keep=0.5).sample(valid)
    assert keep.mean().item() == pytest.approx(0.5, abs=0.01)
    rates = RandomSelector(mode="tokens").sample(valid).mean(-1)
    # Uniform on [0, 1] per example, so a fifth of the rows keep under 20%.
    assert (rates < 0.2).float().mean().item() == pytest.approx(0.2, abs=0.03)


def test_spans_keep_one_contiguous_run_of_the_drawn_share():
    th.manual_seed(0)
    valid = valid_rows([10, 7, 3] * 100)
    keep = RandomSelector(mode="spans", min_keep=0.3, max_keep=0.3).sample(valid)
    for row, mask in zip(valid, keep):
        kept = mask.nonzero().flatten().tolist()
        assert kept == list(range(kept[0], kept[0] + len(kept)))
        assert len(kept) == math.ceil(0.3 * int(row.sum()))
        assert kept[-1] < row.sum()


def test_a_full_keep_rate_keeps_every_valid_position():
    valid = valid_rows([12, 4])
    for mode in ("tokens", "spans"):
        keep = RandomSelector(mode=mode, min_keep=1.0, max_keep=1.0).sample(valid)
        assert th.equal(keep, valid)


@pytest.mark.parametrize(
    "arguments", [{"mode": "words"}, {"min_keep": 0.6, "max_keep": 0.4}]
)
def test_the_selector_refuses_settings_it_cannot_honour(arguments):
    with pytest.raises(ValueError):
        RandomSelector(**arguments)


def test_the_selector_reads_no_states():
    with pytest.raises(NotImplementedError):
        RandomSelector()(th.zeros(1, 3, 4))


def test_the_evaluator_needs_a_random_selector():
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    with pytest.raises(ValueError, match="RandomSelector"):
        Registry.from_key(GRU_EVALX, selectors=MLP_SELECTOR)


class Stub:
    """Predicts class 1 when at least two positions are kept, else class 0."""

    select_over = "word"

    def eval(self):
        return self

    def selection_valid(self, batch):
        return batch.mask

    def predict(self, batch, mask):
        kept = mask.sum(-1)
        # Logits this far apart make the softmax certain.
        return 50 * th.stack([(kept < 2).float(), (kept >= 2).float()], dim=-1)


def frame(rows, run="m", seed="1"):
    return pd.DataFrame(
        [
            {
                "run": run,
                "seed": seed,
                "sample_id": sample_id,
                "tokens": list("abcd"),
                "selected": selected,
                "highlights": highlights,
                "label": label,
                "predicted": predicted,
            }
            for sample_id, selected, highlights, label, predicted in rows
        ]
    )


def test_readings_score_selections_and_both_references():
    batches = [
        SimpleNamespace(sample_ids=th.tensor([3, 4]), mask=th.ones(2, 4)),
        SimpleNamespace(sample_ids=th.tensor([5, 9]), mask=th.ones(2, 4)),
    ]
    readings = evaluator_readings(
        frame(
            [
                (
                    3,
                    [0],
                    [0],
                    1,
                    1,
                ),  # one kept: read as 0, wrong, and so is its highlight
                (4, [0, 1], None, 1, 0),  # two kept: read as 1, right
                (5, [0, 1, 7], [2], 0, 0),  # 7 is past the width, so two kept
            ]
        ),
        Stub(),
        batches,
    ).iloc[0]
    assert readings["documents"] == 3
    assert readings["on_selection"] == pytest.approx(1 / 3)
    assert readings["agrees"] == pytest.approx(0 / 3)
    assert readings["on_full"] == pytest.approx(2 / 3)
    assert readings["annotated"] == 2
    assert readings["on_highlight"] == pytest.approx(1 / 2)
    # The stub is certain, so each probability equals its accuracy.
    for name in ("selection", "full", "highlight"):
        assert readings[f"p_{name}"] == pytest.approx(readings[f"on_{name}"])


def test_readings_keep_runs_and_seeds_apart():
    batches = [SimpleNamespace(sample_ids=th.tensor([1]), mask=th.ones(1, 4))]
    rows = pd.concat(
        [frame([(1, [0], None, 0, 0)]), frame([(1, [0, 1], None, 0, 1)], seed="2")]
    )
    readings = evaluator_readings(rows, Stub(), batches).set_index("seed")
    assert readings.loc["1", "on_selection"] == 1.0
    assert readings.loc["2", "on_selection"] == 0.0
    assert pd.isna(readings.loc["2", "on_highlight"])


def test_readings_need_an_evaluator_that_selects_over_words():
    stub = Stub()
    stub.select_over = "subtoken"
    with pytest.raises(ValueError, match="words"):
        evaluator_readings(frame([(1, [0], None, 0, 0)]), stub, [])


def test_the_analyzer_reads_a_toy_run_with_a_trained_evaluator(tmp_path):
    Registry.build(directory=Path(pyhighlights.__file__).parent)
    short = {"accelerator": "cpu", "max_epochs": 1}
    Registry.from_key(
        TOY_TASK,
        save_path=str(tmp_path / "models"),
        seeds=[0],
        store_predictions=True,
        trainer_args=short,
    ).run()
    Registry.from_key(
        TOY_EVALX_TASK,
        save_path=str(tmp_path / "evaluators"),
        seeds=[0, 1],
        trainer_args=short,
    ).run()
    (evaluator,) = (tmp_path / "evaluators" / "toy-evalx").iterdir()

    readings = SelectionCodeAnalyzer(tmp_path / "models").evaluated(evaluator)

    assert sorted(readings["evaluator_seed"]) == ["0", "1"]
    assert (readings["documents"] > 0).all()
    for column in ("on_selection", "p_selection", "agrees", "on_full", "on_highlight"):
        assert readings[column].between(0, 1).all()
