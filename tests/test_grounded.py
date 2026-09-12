"""Grounded SPP: a highlight pair per knowledge entry, and the subset named.

The decisions these pin are in ``DESIGN_knowledge_grounding``. Two of them are
load-bearing and neither is visible from the shapes: the union the predictor
reads is **ungated**, and the gate is **not** repaired when it comes out empty.
"""

from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.data import (
    HighlightCollator,
    HighlightExample,
    VocabularyTokenizer,
)
from pyhighlights.components.models.base import InputData
from pyhighlights.configurations.keys import GRU_GROUNDED

VOCABULARY = {word: index + 1 for index, word in enumerate("a b c d e f".split())}
ENTRIES = [["a", "b"], ["c", "d"], ["e"]]


def build_registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


def collate(examples, knowledge_size=None):
    collator = HighlightCollator(
        VocabularyTokenizer(VOCABULARY), knowledge_size=knowledge_size
    )
    return collator(examples)


def grounded(**kwargs):
    build_registry()
    th.manual_seed(0)
    model = Registry.from_key(GRU_GROUNDED, **kwargs)
    model.load_knowledge(
        collate(
            [HighlightExample(index, tokens, 0) for index, tokens in enumerate(ENTRIES)]
        )
    )
    return model


def batch():
    return collate(
        [
            HighlightExample(0, ["a", "b", "c", "d"], 1, knowledge=[0, 1]),
            HighlightExample(1, ["e", "f"], 0, knowledge=[]),
        ],
        knowledge_size=len(ENTRIES),
    )


def test_a_selector_reads_the_states_and_the_partner_beside_them():
    """Conditioning is a concatenation, which is why there is no new contract.

    The partner's pooled state is appended to the states being scored, so an
    ordinary ``SPPSelector`` becomes a conditioned one at twice the width and
    every registered implementation works unchanged.
    """
    model = grounded()

    assert model.selector_input_size(model.selector_backbone) == (
        2 * model.selector_backbone.output_size
    )
    assert model.selectors[0].selector[0].in_features == (
        2 * model.selector_backbone.output_size
    )


def test_a_grounded_model_without_a_knowledge_base_says_so():
    build_registry()
    model = Registry.from_key(GRU_GROUNDED)

    with pytest.raises(ValueError, match="no knowledge base was loaded"):
        model.knowledge()


def test_the_output_keeps_one_highlight_pair_per_entry():
    """The experts' object: what was proposed against each entry, not the union.

    A lawyer reading one clause wants the highlight the model drew against
    every rationale, and for a clause nothing explains that is the only way to
    see why none of them fired.
    """
    model = grounded()
    data = batch()
    model.train()

    output = model(data)
    entries, width = len(ENTRIES), data.mask.shape[1]
    entry_width = model.knowledge().mask.shape[1]

    assert output.pair_highlight_mask.shape == (2, 1, entries, width)
    assert output.knowledge_highlight_mask.shape == (2, 1, entries, entry_width)
    assert output.knowledge_logits.shape == (2, 1, entries, 2)
    assert output.knowledge_mask.shape == (2, 1, entries)
    # The head axis every other SPP output carries, so the aggregator, the
    # metrics and `unbind` treat a grounded output as an ordinary one.
    assert output.class_logits.shape[1] == 1
    assert output.highlight_mask.shape == (2, 1, width)


def test_the_union_the_predictor_reads_is_not_gated():
    """The decision the model rests on, and the one shapes cannot show.

    Gating the union would leave an example that instantiates nothing with an
    empty highlight, which the token-level repair rescues to exactly one word.
    Every negative example would then carry a one-word highlight and the
    predictor would learn that one word means negative -- the label decided by
    the size of the selection. Ungated, a negative example carries a full-size
    highlight that can be read beside a positive one.
    """
    model = grounded()
    data = batch()
    # Read at evaluation: `select_activation` is an argmax there, where in
    # training it samples and a shut gate would reopen now and then.
    model.eval()

    output = model(data)
    head = next(output.unbind(dim=1))

    assert th.equal(head.highlight_mask, head.pair_highlight_mask.amax(dim=1))

    # Force every gate shut: the union must not move.
    with th.no_grad():
        for parameter in model.comparer.parameters():
            parameter.zero_()
        model.comparer.comparer[-1].bias.copy_(th.tensor([10.0, -10.0]))
    shut = next(model(data).unbind(dim=1))

    assert shut.knowledge_mask.sum() == 0
    assert shut.highlight_mask.sum(dim=1).min() > 1


def test_an_empty_knowledge_set_is_kept_rather_than_repaired():
    """``K_x = {}`` is the right answer for an example nothing explains.

    The token axis repairs an empty selection, because a predictor handed
    nothing learns a constant. The knowledge axis must not: repairing it would
    invent a reason for every example that has none, and those are the
    majority of the corpora this is for.
    """
    model = grounded()
    data = batch()
    model.eval()

    with th.no_grad():
        for parameter in model.comparer.parameters():
            parameter.zero_()
        model.comparer.comparer[-1].bias.copy_(th.tensor([10.0, -10.0]))
    head = next(model(data).unbind(dim=1))

    assert head.knowledge_mask.sum() == 0
    # Every pair still selected something, which is what keeps the union whole.
    assert head.pair_highlight_mask.sum(dim=-1).min() >= 1


def test_repair_keeps_one_position_per_pair_not_one_per_sample():
    """The repair runs on the last axis, so a pair axis in front is fine.

    ``select`` used to write this inline over ``dim=1``. Grounded models repair
    one selection per ``(sample, entry)`` pair, and both go through the same
    code so the two cannot drift.
    """
    model = grounded()
    logits = th.zeros(2, 3, 4, 2)
    logits[..., 1] = th.tensor([1.0, 5.0, 2.0, 0.0])
    valid = th.ones(2, 3, 4)

    repaired = model.repair_empty(logits, th.zeros(2, 3, 4), valid)

    assert repaired.shape == (2, 3, 4)
    # `hard + scores - scores.detach()` keeps the gradient path open and
    # cancels to one only up to floating point.
    assert th.allclose(repaired.sum(dim=-1), th.ones(2, 3))
    assert th.equal(repaired.argmax(dim=-1), th.ones(2, 3, dtype=th.long))


def test_the_links_train_the_comparer(tmp_path):
    """One step moves the gate towards the gold links, and the loop closes.

    The comparer is what the gold annotation supervises. Without that term it
    learns how often each entry fires and stops reading its inputs, which is
    the degeneracy the memory-network treatment of this corpus needed strong
    supervision to avoid.
    """
    model = grounded()
    data = batch()
    model.train()

    before = model(data)
    loss, terms = model.compute_loss(input_data=data, output_data=before)

    assert "knowledge" in terms
    assert th.isfinite(loss)

    optimizer = th.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(20):
        optimizer.zero_grad()
        output = model(data)
        step_loss, _ = model.compute_loss(input_data=data, output_data=output)
        step_loss.backward()
        optimizer.step()

    model.eval()
    with th.no_grad():
        scored = next(model(data).unbind(dim=1)).knowledge_logits

    # The annotated example instantiates entries 0 and 1 and not entry 2; the
    # unannotated one is skipped by the criterion and is not asserted on.
    linked = scored[0, :, 1] - scored[0, :, 0]
    assert linked[0] > linked[2]
    assert linked[1] > linked[2]


def test_an_unannotated_row_is_skipped_rather_than_scored_as_empty():
    """``-1`` is not ``0``, and the criterion has to keep them apart.

    A corpus that annotates no links at all must contribute nothing to this
    term rather than teaching the comparer that nothing ever applies.
    """
    model = grounded()
    unannotated = collate(
        [HighlightExample(0, ["a", "b", "c", "d"], 1)],
        knowledge_size=len(ENTRIES),
    )
    model.train()

    _, terms = model.compute_loss(
        input_data=unannotated, output_data=model(unannotated)
    )

    assert terms["knowledge"] == 0


def test_faithfulness_is_measured_on_both_axes():
    """The token terms every SPP model has, plus the two the base adds.

    Rationale comprehensiveness is what the pipeline stands or falls on: a
    model that predicts the same thing when the entries it named are taken
    away has grounding that is decoration.
    """
    model = grounded()
    data = batch()
    model.eval()

    with th.no_grad():
        terms = model.faithfulness(data, model(data))

    assert set(terms) == {
        "sufficiency",
        "comprehensiveness",
        "rationale_sufficiency",
        "rationale_comprehensiveness",
    }
    assert all(value.shape == (2,) for value in terms.values())


def test_the_rationale_ablation_is_over_entries_not_over_a_gate():
    """Restricting the base restricts which pairs enter the union.

    The union is ungated, so `K` against `K \\ K_x` is a different set of
    pairs rather than a gate switched off, and the two ablations have to
    partition the pairs between them.
    """
    model = grounded()
    data = batch()
    model.eval()

    with th.no_grad():
        head = next(model(data).unbind(dim=1))

    gate = head.knowledge_mask.unsqueeze(-1)
    pairs = head.pair_highlight_mask
    named = (pairs * gate).amax(dim=1)
    rest = (pairs * (1 - gate)).amax(dim=1)

    assert th.equal(th.maximum(named, rest), head.highlight_mask)


def test_the_base_follows_the_module_onto_its_device():
    """It is moved, not registered: a checkpoint should not carry the corpus."""
    model = grounded()

    assert "_knowledge" not in dict(model.named_buffers())
    assert not any("_knowledge" in name for name in model.state_dict())
    assert isinstance(model.knowledge(), InputData)


def test_a_grounded_model_trains_through_a_real_loop(tmp_path):
    """The acceptance criterion: the whole loop closes, not just the forward.

    A task builds the base, the model is moved to its device with the base
    following, Lightning steps it, the metrics read a grounded output through
    the aggregator, and a run is written down. The toy corpus annotates no
    links, so the knowledge term contributes nothing here -- what this proves
    is the plumbing, and the term itself is scored above.
    """
    from pyhighlights.components.tasks import SPPTask
    from pyhighlights.configurations.keys import TOY

    build_registry()

    class Grounded(SPPTask):
        def knowledge(self):
            return ENTRIES

    task = Grounded(
        loader=TOY,
        model=GRU_GROUNDED,
        save_path=str(tmp_path),
        seeds=[0],
        batch_size=4,
        trainer_args={"max_epochs": 1},
    )
    result = task.run()

    assert result["runs"], "a grounded run wrote nothing down"


def test_link_metrics_score_the_set_and_the_links_separately():
    """Four numbers, because any one of them would hide the others.

    Two entries, chosen so the two averages disagree. Entry 0 is found every
    time; entry 1 is missed once and invented once. Micro weights every link
    equally and reports 0.67; macro averages over entries, and the entry that
    was failed drags it to 0.50. On a real base the failed entry is often the
    rare one that decides a case, which is why both are reported.
    """
    build_registry()
    from pyhighlights.configurations.keys import (
        EMPTY_SET_METRIC,
        EXACT_SET_METRIC,
        LINK_F1_METRIC,
        LINK_MACRO_F1_METRIC,
    )

    values = {
        "knowledge_mask": th.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
        "knowledge_true": th.tensor([[1, 0], [1, 1], [0, 0]]),
    }

    scores = {}
    for metric_key in (
        LINK_F1_METRIC,
        LINK_MACRO_F1_METRIC,
        EXACT_SET_METRIC,
        EMPTY_SET_METRIC,
    ):
        bound = Registry.from_key(metric_key)
        bound.update(values)
        scores[bound.name] = float(bound.compute())

    assert scores["link_f1"] == pytest.approx(2 / 3)
    assert scores["link_macro_f1"] == pytest.approx(0.5)
    # Only the first example names exactly its annotated set.
    assert scores["exact_set_match"] == pytest.approx(1 / 3)
    # The one example that instantiates nothing was grounded in entry 1 anyway.
    assert scores["empty_set_accuracy"] == pytest.approx(0.0)


def test_an_unannotated_example_is_skipped_by_the_set_metrics():
    """``-1`` is not an empty set, and scoring it as one would flatter."""
    from pyhighlights.utility.metrics import EmptySetAccuracy, ExactSetMatch

    preds = th.tensor([[0.0, 0.0], [0.0, 0.0]])
    target = th.tensor([[-1, -1], [0, 0]])

    for metric in (ExactSetMatch(), EmptySetAccuracy()):
        metric.update(preds, target)
        assert float(metric.examples) == 1


def test_a_positive_weight_per_entry_is_what_a_shared_one_cannot_do():
    """The reason knowledge supervision needs a binary criterion.

    Two classes under a cross entropy carry one positive weight. The knowledge
    axis needs a different weight per entry, since the entry that decides a
    case is frequently the one that fires on a handful of examples.
    """
    from pyhighlights.utility.losses import MaskedBinaryCrossEntropy

    scores = th.zeros(2, 3)
    gold = th.tensor([[1, 0, 0], [0, 0, 1]])
    valid = th.ones(2, 3)

    flat = MaskedBinaryCrossEntropy()(scores, gold, valid)
    weighted = MaskedBinaryCrossEntropy(pos_weight=[1.0, 1.0, 50.0])(
        scores, gold, valid
    )

    assert weighted > flat

    with pytest.raises(ValueError, match="pos_weight has 2 entries"):
        MaskedBinaryCrossEntropy(pos_weight=[1.0, 1.0])(scores, gold, valid)


def test_the_binary_criterion_skips_what_nobody_annotated():
    """A ``-1`` reaching a binary target is a number, not a label."""
    from pyhighlights.utility.losses import MaskedBinaryCrossEntropy

    scores = th.zeros(2, 2)
    valid = th.ones(2, 2)
    unannotated = MaskedBinaryCrossEntropy()(
        scores, th.tensor([[-1, -1], [-1, -1]]), valid
    )

    assert unannotated == 0


def test_the_supervision_term_reads_the_score_the_gate_came_from():
    """One quantity, so the term and the gate cannot disagree."""
    from pyhighlights.configurations.keys import KNOWLEDGE_SUPERVISION_LOSS

    model = grounded(losses=[KNOWLEDGE_SUPERVISION_LOSS])
    data = batch()
    model.train()

    output = model(data)
    head = next(output.unbind(dim=1))
    loss, terms = model.compute_loss(input_data=data, output_data=output)

    assert th.allclose(
        head.knowledge_score,
        head.knowledge_logits[..., 1] - head.knowledge_logits[..., 0],
    )
    assert "knowledge" in terms
    assert th.isfinite(loss)


def test_a_span_count_separates_two_phrases_from_eight_fragments():
    """A rate cannot, and a lawyer reads phrases.

    Twenty per cent of a clause in two spans is readable; the same share
    scattered over eight is not, and they have identical selection rates.
    """
    from pyhighlights.components.analyzers import spans

    assert spans([]) == 0
    assert spans([3, 4, 5]) == 1
    assert spans([0, 1, 5, 6]) == 2
    assert spans([6, 5, 1, 0]) == 2


def test_readability_is_reported_per_class_not_pooled():
    """Domain experts asked whether negative highlights differ from positive.

    A pooled average over a split that is overwhelmingly negative reports the
    negative examples' number and calls it the model's.
    """
    import pandas as pd

    from pyhighlights.components.analyzers import readability

    frame = pd.DataFrame(
        {
            "run": ["r", "r", "r"],
            "label": [0, 1, 1],
            "predicted": [0, 1, 1],
            "tokens": [["a", "b", "c", "d"]] * 3,
            "selected": [[0], [0, 1], [0, 1, 3]],
        }
    )
    report = readability(frame)

    assert len(report) == 2
    by_label = {
        int(row): value
        for row, value in zip(report["label"], report[("selection_size", "mean")])
    }
    assert by_label[0] == pytest.approx(1.0)
    assert by_label[1] == pytest.approx(2.5)
    assert readability(pd.DataFrame()).empty


def test_knowledge_weights_are_read_off_the_split_not_typed():
    """One positive weight per entry, and a refusal where there is none.

    An entry no example links to has no frequency to invert, and both a zero
    and an infinity would train something the corpus never showed -- the same
    refusal `class_weights` already makes.
    """
    import pandas as pd

    from pyhighlights.components.preprocessors import KnowledgeWeights, link_weights

    # Entry 0 fires in one annotated example of three, entry 1 in two.
    assert link_weights([[0], [1], [1], None], entries=2) == [2.0, 0.5]

    frame = pd.DataFrame({"knowledge": [[0], [1], [1], None]})
    weights = KnowledgeWeights(entries=2)
    weights.process({"train": frame})

    assert weights.weights == [2.0, 0.5]
    assert weights.counts == {0: 1, 1: 2}

    with pytest.raises(ValueError, match="entries \\[2\\] are linked by no example"):
        link_weights([[0], [1]], entries=3)
    with pytest.raises(KeyError, match="no `knowledge` column"):
        KnowledgeWeights(entries=2).process({"train": pd.DataFrame({"label": [0]})})
