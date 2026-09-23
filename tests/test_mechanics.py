"""What each step of a model does, checked one step at a time.

The suite elsewhere checks that a model trains, that a key resolves and that a
number comes out. None of that catches a mask applied to the wrong axis or a
module handed the wrong tensor: a model with a leaking bottleneck still trains
and still reports an F1. ``StackedBackbone`` fed dropped words to its recurrent
encoder for four releases and every test passed.

So these are mechanical. Each one names a single step -- what the encoder
attends over, what the selector scores, what the predictor is handed, which
axis a fold runs along -- and asserts what that step must be true of, with
tensors small enough to reason about by hand. They are the tests worth having
before a run that costs GPU-days, because the failures they catch are silent.

The pattern throughout: change one thing that must not matter, and assert the
output does not move. That is stronger than checking a shape and it is what a
contract actually claims.
"""

from pathlib import Path

import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components.data import (
    HighlightCollator,
    HighlightExample,
    TokenizedExample,
    VocabularyTokenizer,
)
from pyhighlights.components.models.spp.base import SPP
from pyhighlights.configurations.keys import GRU_BACKBONE, GRU_FR

VOCABULARY = {word: index + 2 for index, word in enumerate("a b c d e f g".split())}


def build_registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


class SubwordTokenizer:
    """Splits ``b`` into two pieces, so the two axes genuinely differ."""

    pad_token_id = 0

    def encode(self, tokens, max_length=None):
        ids, words = [], []
        for index, token in enumerate(tokens):
            pieces = [20, 21] if token == "b" else [VOCABULARY.get(token, 0)]
            ids += pieces
            words += [index] * len(pieces)
        if max_length is not None:
            ids, words = ids[:max_length], words[:max_length]
        return TokenizedExample(ids, words)


def model(**kwargs):
    build_registry()
    return Registry.from_key(GRU_FR, **kwargs)


def batch(examples, tokenizer=None, **kwargs):
    return HighlightCollator(tokenizer or VocabularyTokenizer(VOCABULARY), **kwargs)(
        examples
    )


# --------------------------------------------------------------------------
# The batch: which axis a field lives on, and what padding means on each.
# --------------------------------------------------------------------------


def test_the_word_axis_is_words_and_the_subtoken_axis_is_subtokens():
    """Two axes, and a field belongs to exactly one of them.

    ``features`` and ``word_ids`` are subtokens, ``mask`` and
    ``highlight_true`` are words. Reading a word-axis field on the subtoken
    axis is off by the number of split words, which for legal text is most of
    them.
    """
    data = batch(
        [HighlightExample(0, ["a", "b", "c"], 1, [1, 0, 1])], SubwordTokenizer()
    )

    # "b" is two subtokens, so four subtokens over three words.
    assert data.features.shape == (1, 4)
    assert data.word_ids.tolist() == [[0, 1, 1, 2]]
    assert data.mask.shape == (1, 3)
    assert data.highlight_true.tolist() == [[1, 0, 1]]


def test_padding_is_excluded_on_both_axes_and_carries_no_annotation():
    data = batch(
        [
            HighlightExample(0, ["a", "b", "c"], 1, [1, 0, 1]),
            HighlightExample(1, ["a"], 0, [0]),
        ]
    )

    assert data.mask.tolist() == [[1, 1, 1], [1, 0, 0]]
    # `-1` and not `0`: a padded position is unannotated, not annotated negative.
    assert data.highlight_true.tolist() == [[1, 0, 1], [0, -1, -1]]
    assert data.attention_mask.tolist() == [[1, 1, 1], [1, 0, 0]]


def test_the_word_axis_stops_where_truncation_did():
    """A word past the cut was never encoded, so it cannot be selected.

    If the word axis stayed as wide as the document, a selector would be
    offered positions the encoder never saw and a sparsity rate would be a
    fraction of a document that was not read.
    """
    data = batch(
        [HighlightExample(0, ["a", "b", "c", "d"], 1)], SubwordTokenizer(), max_length=3
    )

    # Three subtokens covers "a" and both pieces of "b" -- two words.
    assert data.features.shape == (1, 3)
    assert data.mask.tolist() == [[1, 1]]


# --------------------------------------------------------------------------
# The encoder: what it attends over, and what it must not see.
# --------------------------------------------------------------------------


def test_the_encoder_attends_over_specials_but_they_are_never_selectable():
    """Two different masks, and conflating them costs either way.

    ``attention`` is what the encoder reads and includes ``[CLS]``; ``mask``
    is what may be selected and does not. Dropping specials from the attention
    moves a pretrained encoder off its distribution; adding them to the
    selection lets a model highlight a token that carries no word.
    """

    class Specials:
        pad_token_id = 0

        def encode(self, tokens, max_length=None):
            return TokenizedExample(
                [101, *range(10, 10 + len(tokens)), 102],
                [None, *range(len(tokens)), None],
            )

    data = batch([HighlightExample(0, ["a", "b"], 1)], Specials())
    spp = model()

    assert data.attention().tolist() == [[1, 1, 1, 1]]
    assert spp.selection_valid(data).tolist() == [[1, 1]]
    assert spp.encoder_mask(data).tolist() == [[1, 1, 1, 1]]


def test_a_special_token_is_unselectable_on_the_subtoken_axis_too():
    """The other half of the guarantee, on the axis where specials exist.

    Selecting over words makes this structural -- a special token belongs to no
    word, so it is not on the axis at all. Selecting over subtokens puts it on
    the axis, and ``selection_valid`` is then the only thing keeping it out. A
    model that could mark ``[CLS]`` would report a highlight containing no word.
    """

    class Specials:
        pad_token_id = 0

        def encode(self, tokens, max_length=None):
            return TokenizedExample(
                [101, *range(10, 10 + len(tokens)), 102],
                [None, *range(len(tokens)), None],
            )

    data = batch([HighlightExample(0, ["a", "b"], 1)], Specials())

    # Four subtokens: [CLS], two words, [SEP]. Only the two words may be
    # selected, and the encoder still attends over all four.
    assert model(select_over="subtoken").selection_valid(data).tolist() == [
        [0.0, 1.0, 1.0, 0.0]
    ]
    assert model(select_over="subtoken").encoder_mask(data).tolist() == [[1, 1, 1, 1]]


def test_a_padded_subtoken_is_unselectable_on_the_subtoken_axis_too():
    """Padding carries no word id either, so the same guard excludes it."""

    class Specials:
        pad_token_id = 0

        def encode(self, tokens, max_length=None):
            return TokenizedExample(
                [101, *range(10, 10 + len(tokens)), 102],
                [None, *range(len(tokens)), None],
            )

    data = batch(
        [HighlightExample(0, ["a", "b"], 1), HighlightExample(1, ["a"], 0)], Specials()
    )

    # The shorter row is padded to the longer one's width; nothing past its
    # own [SEP] is selectable.
    assert model(select_over="subtoken").selection_valid(data).tolist() == [
        [0.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ]


def test_padding_never_reaches_the_encoders_output():
    build_registry()
    backbone = Registry.from_key(GRU_BACKBONE, hidden_size=4)
    data = batch(
        [HighlightExample(0, ["a", "b", "c"], 1), HighlightExample(1, ["a"], 0)]
    )

    states = backbone.encode(data.features, data.mask)

    assert not states[~data.mask.bool()].any()


def test_a_dropped_word_cannot_change_what_the_predictor_reads():
    """The bottleneck, at the step that enforces it.

    This is the shape of the bug `StackedBackbone` carried: masking a token out
    of the attention is not the same as keeping its state out of the encoder's
    output. Checked on the pooled vector because that is what the predictor is
    actually handed.
    """
    build_registry()
    backbone = Registry.from_key(GRU_BACKBONE, hidden_size=4)
    backbone.eval()

    mask = th.ones(1, 5)
    selection = th.tensor([[1.0, 1.0, 0.0, 0.0, 1.0]])
    before = th.tensor([[1, 2, 3, 4, 5]])
    after = before.clone()
    after[0, 2], after[0, 3] = 6, 7

    with th.no_grad():
        pooled = [
            backbone.pool(backbone.encode(features, mask, selection), mask * selection)
            for features in (before, after)
        ]

    assert th.allclose(pooled[0], pooled[1], atol=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="Known, measured, and not yet fixed: the predictor can read the "
    "*shape* of the mask as well as the words it kept. A GRU steps its "
    "recurrence over dropped positions with a zero input, so the number of "
    "them changes the state; a transformer gives a kept word a different "
    "position embedding when the gap before it changes. Compaction -- "
    "gathering the kept positions instead of zeroing the dropped ones -- is "
    "the candidate fix.",
)
def test_the_gap_between_kept_words_cannot_change_what_the_predictor_reads():
    """The same highlight must mean the same input, wherever its words sat.

    ``test_a_dropped_word_cannot_change_what_the_predictor_reads`` changes what
    a dropped word *is*. This changes how many there are. Both are outside the
    highlight, so under `the highlight is the predictor's input` neither may
    move the prediction -- but only the first was ever checked, and the second
    is the channel by which a selector can signal a label through the count.

    Two rows, identical kept words in the same order, different gaps between
    them. A model whose predictor reads its highlight and nothing else answers
    the same for both.
    """
    build_registry()
    backbone = Registry.from_key(GRU_BACKBONE, hidden_size=8)
    backbone.eval()

    mask = th.ones(2, 7)
    #            kept  drop  drop  kept  pad-ish filler
    features = th.tensor([[2, 9, 9, 3, 9, 9, 9], [2, 3, 9, 9, 9, 9, 9]])
    selection = th.tensor([[1.0, 0, 0, 1.0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0, 0]])

    with th.no_grad():
        pooled = backbone.pool(
            backbone.encode(features, mask, selection), mask * selection
        )

    assert th.allclose(pooled[0], pooled[1], atol=1e-6)


def test_compaction_moves_the_kept_words_to_the_front_in_order():
    """The primitive, on tensors small enough to read."""
    features = th.tensor([[2, 9, 9, 3, 9, 9, 9], [2, 3, 9, 9, 9, 9, 9]])
    keep = th.tensor([[1.0, 0, 0, 1.0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0, 0]])

    compacted, mask = SPP.compacted(features, keep)

    # Same kept words, same order, and the gap is gone from both rows.
    assert compacted.tolist() == [[2, 3], [2, 3]]
    assert mask.tolist() == [[1.0, 1.0], [1.0, 1.0]]


def test_compaction_cuts_to_the_widest_selection_in_the_batch():
    """A row that kept fewer is padded, not stretched."""
    features = th.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    keep = th.tensor([[1.0, 1.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0]])

    compacted, mask = SPP.compacted(features, keep)

    assert compacted[0].tolist() == [2, 3, 4]
    assert mask.tolist() == [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]


def test_compaction_keeps_the_gradient_path_to_the_selector_open():
    """The permutation is detached; the mask values are not.

    Gathering is a reordering rather than a quantity, so the gradient runs
    through the mask values that come back, not through the argsort.
    """
    features = th.tensor([[2, 9, 9, 3]])
    keep = th.tensor([[1.0, 0.0, 0.0, 1.0]], requires_grad=True)

    _, mask = SPP.compacted(features, keep)
    mask.sum().backward()

    assert keep.grad is not None
    assert keep.grad.abs().sum() > 0


def test_a_compact_model_answers_the_same_for_two_gaps_of_one_highlight():
    """End to end, through ``SPP.predict``, which is what a run uses.

    The tests above check the primitive. This checks the flag: two clauses
    whose highlights hold the same words in the same order, differing only in
    what sits between them, must reach the predictor as one input.
    """
    spp = model(compact=True)
    spp.eval()
    data = batch(
        [
            HighlightExample(0, ["a", "b", "c", "d"], 1),
            HighlightExample(1, ["a", "d", "b", "c"], 1),
        ]
    )
    # `a` and `d` in both, adjacent in the second and two apart in the first.
    highlight = th.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    with th.no_grad():
        logits = spp.predict(data, highlight)

    assert th.allclose(logits[0], logits[1], atol=1e-6)


def test_a_model_without_compaction_does_not_answer_the_same():
    """The control: the difference above is the flag and not the fixture."""
    spp = model(compact=False)
    spp.eval()
    data = batch(
        [
            HighlightExample(0, ["a", "b", "c", "d"], 1),
            HighlightExample(1, ["a", "d", "b", "c"], 1),
        ]
    )
    highlight = th.tensor([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    with th.no_grad():
        logits = spp.predict(data, highlight)

    assert not th.allclose(logits[0], logits[1], atol=1e-6)


def test_a_compact_model_still_trains_its_selector():
    """The load-bearing claim, checked on a model rather than on the primitive.

    Compaction gathers by an index derived from the mask, and an index is not
    differentiable. If the gradient went with it the selector would stop
    learning and the arm would look like a result about compaction when it was
    a result about a dead optimizer.
    """
    data = batch(
        [
            HighlightExample(0, ["a", "b", "c", "d"], 1),
            HighlightExample(1, ["b", "c"], 0),
        ]
    )

    grads = {}
    for flag in (False, True):
        th.manual_seed(0)
        spp = model(compact=flag)
        spp.train()
        spp(data).class_logits.sum().backward()
        selector = [p for name, p in spp.named_parameters() if "selector" in name]
        assert all(p.grad is not None for p in selector), flag
        grads[flag] = sum(float(p.grad.abs().sum()) for p in selector)

    assert grads[True] > 0
    # Not a claim that the two are equal -- they are different computations --
    # only that compaction has not collapsed the signal by an order of
    # magnitude, which is what a severed path would look like.
    assert grads[True] > grads[False] / 10


def test_compaction_keeps_a_special_token_where_a_pretrained_encoder_expects_it():
    """`[CLS]` first and `[SEP]` last, after the gaps are gone.

    `to_subtokens` always keeps a special token, and the gather is stable, so
    the two ends stay the two ends. A transformer pretrained on that shape
    would otherwise be handed a sequence starting mid-clause.
    """
    features = th.tensor([[101, 2000, 3000, 4000, 5000, 102]])
    keep = th.tensor([[1.0, 0.0, 0.0, 1.0, 0.0, 1.0]])

    compacted, _ = SPP.compacted(features, keep)

    assert compacted.tolist() == [[101, 4000, 102]]


def test_compaction_rounds_the_width_rather_than_truncating_it():
    """A mask that is not exactly binary must not lose a kept position.

    Every mask reaching `compacted` is 0.0 or 1.0 in the forward pass today.
    `int()` on a sum that is not would be a silent off-by-some: two positions
    at 0.9 sum to 1.8 and truncate to a width of one.
    """
    features = th.tensor([[2, 3, 4, 5]])
    soft = th.tensor([[0.9, 0.9, 0.0, 0.0]])

    compacted, _ = SPP.compacted(features, soft)

    assert compacted.shape[1] == 2


def test_compaction_survives_a_batch_with_no_rows():
    """An empty batch has no widest selection to take the maximum of."""
    compacted, mask = SPP.compacted(th.zeros(0, 5, dtype=th.long), th.zeros(0, 5))

    assert compacted.shape == (0, 1)
    assert mask.shape == (0, 1)


def test_a_compact_complement_is_still_the_complement():
    """Compaction gathers whichever side it is given, and they stay disjoint.

    `predict_complement` passes `valid * (1 - highlight)`, so under compaction
    the predictor reads the dropped words gathered together. The two passes
    have to stay different inputs, or comprehensiveness would be measuring one
    thing twice.
    """
    features = th.tensor([[2, 3, 4, 5]])
    highlight = th.tensor([[1.0, 0.0, 0.0, 1.0]])

    kept, _ = SPP.compacted(features, highlight)
    complement, _ = SPP.compacted(features, 1.0 - highlight)

    assert kept.tolist() == [[2, 5]]
    assert complement.tolist() == [[3, 4]]


def test_compaction_closes_the_gap_channel_on_every_backbone(monkeypatch):
    """What the xfail tests above are waiting for, under ``compact=True``.

    Same kept words, same order, different gaps. With the dropped positions
    zeroed in place the predictor's input moves; with them gathered away it
    does not, because the two rows become the same sequence.
    """
    build_registry()
    features = th.tensor([[2, 9, 9, 3, 9, 9, 9], [2, 3, 9, 9, 9, 9, 9]])
    keep = th.tensor([[1.0, 0, 0, 1.0, 0, 0, 0], [1.0, 1.0, 0, 0, 0, 0, 0]])
    compacted, mask = SPP.compacted(features, keep)

    backbone = Registry.from_key(GRU_BACKBONE, hidden_size=8)
    backbone.eval()
    with th.no_grad():
        pooled = backbone.pool(backbone.encode(compacted, mask), mask)

    assert th.allclose(pooled[0], pooled[1], atol=1e-6)


# --------------------------------------------------------------------------
# The fold between axes: which direction, and along which axis.
# --------------------------------------------------------------------------


def test_states_fold_to_words_by_mean_and_a_distribution_by_sum():
    """A vector is averaged over a word's pieces; a share is added up.

    Averaging a distribution would report a long word as less attended than the
    short one beside it, purely for being spelled with more subtokens. That is
    the bug G-RAT's guider had.
    """
    spp = model()
    data = batch([HighlightExample(0, ["a", "b", "c"], 1)], SubwordTokenizer())
    states = th.tensor([[[1.0], [2.0], [4.0], [8.0]]])

    assert spp.to_words(states, data).squeeze(-1).tolist() == [[1.0, 3.0, 8.0]]
    assert spp.to_words(states, data, reduce="sum").squeeze(-1).tolist() == [
        [1.0, 6.0, 8.0]
    ]


def test_a_word_is_selected_in_every_piece_of_itself_or_in_none():
    """What makes an exported highlight the predictor's actual input.

    Selecting over words and spreading back means a split word is kept whole.
    A subtoken-level selection could keep ``un`` and drop ``##fair``, and the
    export would print the word ``unfair``.
    """
    spp = model()
    data = batch([HighlightExample(0, ["a", "b", "c"], 1)], SubwordTokenizer())

    spread = spp.to_subtokens(th.tensor([[1.0, 0.0, 1.0]]), data)

    # "b" is positions 1 and 2, dropped together.
    assert spread.tolist() == [[1.0, 0.0, 0.0, 1.0]]


def test_the_annotation_is_spread_onto_the_axis_the_selection_is_made_over():
    """A loss binds one field name and scores the unit the model selected in."""
    subtoken = model(select_over="subtoken")
    data = batch(
        [HighlightExample(0, ["a", "b", "c"], 1, [1, 0, 1])], SubwordTokenizer()
    )

    assert subtoken.selection_truth(data).tolist() == [[1, 0, 0, 1]]
    assert model().selection_truth(data).tolist() == [[1, 0, 1]]


# --------------------------------------------------------------------------
# The selector and the predictor: what each is handed.
# --------------------------------------------------------------------------


def test_the_selector_never_marks_a_padded_position():
    spp = model()
    spp.eval()
    data = batch(
        [HighlightExample(0, ["a", "b", "c"], 1), HighlightExample(1, ["a"], 0)]
    )

    with th.no_grad():
        output = spp(data)

    head = next(output.unbind(dim=1))
    assert not head.highlight_mask[~data.mask.bool()].any()


def test_an_empty_selection_is_repaired_to_exactly_one_position():
    """A predictor handed nothing learns a constant, so one token is kept.

    Exactly one, and the highest-scoring one -- a repair that kept two would
    be a sparsity floor nobody configured.
    """
    spp = model()
    logits = th.zeros(2, 4, 2)
    logits[..., 1] = th.tensor([1.0, 5.0, 2.0, 0.0])
    valid = th.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.0, 0.0]])

    repaired = spp.repair_empty(logits, th.zeros(2, 4), valid)

    assert th.allclose(repaired.sum(dim=-1), th.ones(2))
    assert repaired.argmax(dim=-1).tolist() == [1, 1]
    # The repair never reaches an invalid position.
    assert not repaired[~valid.bool()].any()


def test_the_repair_keeps_the_gradient_path_to_the_selector_open():
    """Straight-through, or a repaired sample teaches the selector nothing."""
    spp = model()
    logits = th.zeros(1, 3, 2, requires_grad=True)
    valid = th.ones(1, 3)

    spp.repair_empty(logits, th.zeros(1, 3), valid).sum().backward()

    assert logits.grad is not None and logits.grad.abs().sum() > 0


def test_the_predictor_reads_the_selection_and_the_complement_reads_the_rest():
    """Two passes that must partition the input between them."""
    spp = model()
    spp.eval()
    data = batch([HighlightExample(0, ["a", "b", "c", "d"], 1)])
    highlight = th.tensor([[1.0, 1.0, 0.0, 0.0]])

    with th.no_grad():
        full = spp.predict_full(data)
        on_highlight = spp.predict(data, highlight)
        on_complement = spp.predict_complement(data, highlight)

    assert full.shape == on_highlight.shape == on_complement.shape == (1, 2)
    # The complement is a different input, so a different answer. Equal logits
    # here would mean the mask reached neither pass.
    assert not th.allclose(on_highlight, on_complement)


def test_a_head_is_scored_against_the_class_it_predicted():
    """Faithfulness anchors on the highlight's own prediction.

    Anchoring on the full input would anchor a select-then-predict model on the
    one pass it is never trained for.
    """
    spp = model()
    spp.eval()
    data = batch(
        [HighlightExample(0, ["a", "b", "c"], 1), HighlightExample(1, ["c", "d"], 0)]
    )

    with th.no_grad():
        terms = spp.faithfulness(data, spp(data))

    assert set(terms) == {"sufficiency", "comprehensiveness"}
    assert all(value.shape == (2,) for value in terms.values())


# --------------------------------------------------------------------------
# The optimizer: which parameters land in which group.
# --------------------------------------------------------------------------


def test_only_backbone_parameters_train_at_the_encoder_rate():
    """``encoder_lr`` is defined by where a parameter sits, not by its history.

    A selector or predictor head initialised from scratch at 2e-5 barely moves;
    a pretrained encoder at 1e-3 is destroyed. Getting the split wrong is
    invisible except as a model that does not learn.
    """
    spp = model(encoder_lr=3e-5)
    optimizer = spp.configure_optimizers()

    rates = {group["lr"] for group in optimizer.param_groups}
    assert 3e-5 in rates
    encoders = spp.encoder_ids()
    for group in optimizer.param_groups:
        inside = {id(p) in encoders for p in group["params"]}
        # A group is all encoder or all head, never mixed -- a mixed group
        # would train half its parameters at the wrong rate.
        assert len(inside) == 1
        assert group["lr"] == (3e-5 if inside.pop() else pytest.approx(1e-3))


def test_a_model_with_nothing_to_optimize_says_so():
    spp = model()
    with pytest.raises(ValueError, match="no parameter to optimize"):
        spp.build_optimizer([])
