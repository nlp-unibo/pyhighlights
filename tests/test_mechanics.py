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
from pyhighlights.configurations.keys import GRU_BACKBONE, GRU_FR

VOCABULARY = {word: index + 1 for index, word in enumerate("a b c d e f g".split())}


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
