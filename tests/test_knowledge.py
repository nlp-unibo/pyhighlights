"""The knowledge axis: links from an example to the entries explaining it.

A corpus may annotate *which* knowledge base entries explain an example while
annotating nothing about which words carry them -- ToS-100 does exactly that,
with legal rationales per unfairness category. These tests pin the two things
that annotation needs: the distinction between "no entry applies" and "nobody
said", and the refusal of a link pointing outside the base it indexes.
"""

from pathlib import Path

import pandas as pd
import pytest
import torch as th
from cinnamon.registry import Registry

import pyhighlights
from pyhighlights.components import (
    HighlightCollator,
    HighlightExample,
    VocabularyTokenizer,
)
from pyhighlights.components.loaders import HighlightLoader, to_examples
from pyhighlights.components.models.base import Model
from pyhighlights.components.tasks import SPPTask
from pyhighlights.configurations.keys import GRU_FR, TOY


def build_registry():
    Registry.build(directory=Path(pyhighlights.__file__).parent)


VOCABULARY = {"fair": 1, "unfair": 2, "clause": 3}


def collator(knowledge_size=None):
    return HighlightCollator(
        VocabularyTokenizer(VOCABULARY, unknown_token_id=0, pad_token_id=0),
        knowledge_size=knowledge_size,
    )


def test_an_empty_link_set_is_not_a_missing_one():
    """The distinction the grounded pipeline is scored on.

    A fair clause instantiates no rationale, and that is a gold answer. A
    clause nobody annotated is a row a loss has to skip. Collapsing the two
    would score every unannotated example as if it had been called fair.
    """
    batch = collator(knowledge_size=3)(
        [
            HighlightExample(0, ["unfair", "clause"], 1, knowledge=[0, 2]),
            HighlightExample(1, ["fair", "clause"], 0, knowledge=[]),
            HighlightExample(2, ["fair", "clause"], 0),
        ]
    )
    assert th.equal(
        batch.knowledge_true,
        th.tensor([[1, 0, 1], [0, 0, 0], [-1, -1, -1]]),
    )


def test_a_corpus_without_a_knowledge_base_collates_as_it_always_has():
    batch = collator()([HighlightExample(0, ["fair", "clause"], 0)])
    assert batch.knowledge_true is None


def test_a_link_outside_the_base_is_refused():
    """The failure this catches is silent everywhere else.

    Links are zero-based line numbers into a file, so inserting a line into a
    knowledge base without re-annotating relabels every entry after it. The
    only symptom is a model that learns the wrong rationale.
    """
    with pytest.raises(ValueError, match="outside a base of 2 entries"):
        collator(knowledge_size=2)(
            [HighlightExample(4, ["unfair"], 1, knowledge=[0, 5])]
        )


def test_links_are_indices_and_are_not_repeated():
    with pytest.raises(ValueError, match="non-negative indices"):
        HighlightExample(0, ["fair"], 0, knowledge=[-1])
    with pytest.raises(ValueError, match="must not repeat"):
        HighlightExample(0, ["fair"], 0, knowledge=[1, 1])


def test_the_links_survive_the_frame_and_the_axis_is_the_base_size():
    """A loader's extra column reaches the batch, and ``M`` comes from the base.

    The width of the knowledge axis is a property of the base, not of the
    batch: an entry no example in this batch links to still has a column.
    """

    class Grounded(HighlightLoader):
        def read(self):
            return {}

        def knowledge(self):
            return [["first", "rationale"], ["second"], ["third"]]

    frame = pd.DataFrame(
        {
            "sample_id": [0, 1],
            "text": ["unfair clause", "fair clause"],
            "tokens": [["unfair", "clause"], ["fair", "clause"]],
            "label": [1, 0],
            "highlights": [None, None],
            "knowledge": [[1], []],
        }
    )
    base = Grounded().knowledge()
    batch = collator(knowledge_size=len(base))(to_examples(frame))

    assert [example.knowledge for example in to_examples(frame)] == [(1,), ()]
    assert batch.knowledge_true.shape == (2, 3)
    assert th.equal(batch.knowledge_true, th.tensor([[0, 1, 0], [0, 0, 0]]))


def test_a_loader_without_a_knowledge_base_says_so():
    class Plain(HighlightLoader):
        def read(self):
            return {}

    assert Plain().knowledge() is None


def test_a_model_that_cannot_read_a_knowledge_base_refuses_one():
    """Same contract as ``load_embeddings``: refuse rather than ignore.

    A model handed a knowledge base it drops on the floor would report itself
    as grounded while classifying from the clause alone.
    """

    class Plain(Model):
        def compute_loss(self, input_data, output_data):
            raise NotImplementedError

    with pytest.raises(NotImplementedError, match="does not read a knowledge base"):
        Model.load_knowledge(Plain.__new__(Plain), data=None)


def test_a_task_tokenizes_the_base_once_with_the_corpus_tokenizer(tmp_path):
    """The base reaches the model as a batch, not as a column of every batch.

    It is shared by every example of a run, so it is encoded once. Tokenizing
    it with the collator the corpus uses is what makes its ids comparable to a
    clause's -- a base read by a second tokenizer would put the two texts in
    different vocabularies.
    """
    build_registry()

    class Grounded(SPPTask):
        def knowledge(self):
            return [["a", "legal", "rationale"], ["another", "one"]]

    task = Grounded(loader=TOY, model=GRU_FR, save_path=str(tmp_path))
    task.loaders(task.splits())

    base = task._knowledge
    assert base.features.shape[0] == 2
    # `sample_id` is the index the links point at, so the order the loader
    # returned is the order the annotation indexes.
    assert th.equal(base.sample_ids, th.tensor([0, 1]))
    assert base.mask.sum().item() == 5


def test_a_model_that_cannot_ground_refuses_a_grounded_corpus(tmp_path):
    """The failure mode this exists to prevent is the quiet one.

    A model handed a knowledge base it has nowhere to put would train exactly
    as an ungrounded run does and report itself as the grounded arm. Same
    lesson as MCD silently discarding highlight supervision.
    """
    build_registry()

    class Grounded(SPPTask):
        def knowledge(self):
            return [["a", "legal", "rationale"]]

    task = Grounded(loader=TOY, model=GRU_FR, save_path=str(tmp_path))
    task.loaders(task.splits())

    with pytest.raises(NotImplementedError, match="does not read a knowledge base"):
        task.build_model()
