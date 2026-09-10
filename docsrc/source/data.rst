Data
====

A batch lives on two axes, and which one a field is on is the whole of this
page.

The **subtoken axis** ``[B, T]`` is what the encoder reads — ``features``,
``attention_mask`` and ``word_ids``. A backbone tokenizes however it likes, so
``T`` is its business: a vocabulary tokenizer gives one position per word, a
subword tokenizer gives more.

The **word axis** ``[B, W]`` is what a person reads and what a selection is
made over — ``mask`` and ``highlight_true``. A word is the unit the corpus
annotates, the unit a sparsity target is a fraction of, and the unit an export
shows. It is also the same unit whichever backbone read the text, which is what
lets two backbones share a table.

``word_ids`` maps between them, ``-1`` where a position belongs to no word. For
a vocabulary tokenizer the axes coincide and it is the identity.

Selecting over words
--------------------

A selection is made over words by default. The alternative —
``select_over="subtoken"`` on any SPP model — lets a model keep ``un`` and drop
``##fair``, and an export then reports the word ``unfair``: a highlight that is
not what the predictor read, in a library whose claim is that it is. Measured
on 16 legal clauses with Legal-BERT, an untrained selector splits 38 of 438
words that way; over words it splits none, by construction.

Pooling happens *after* the encoder, never before, so a pretrained backbone
still attends over its own subtokens. What the predictor is then given is the
selection spread back over every piece of each word it kept.

``subtoken`` is kept so the difference can be measured rather than argued
about.

Special tokens
--------------

``[CLS]`` and ``[SEP]`` are added by default. They carry no word, so
``word_ids`` is ``-1`` there and a selector never sees them — but the encoder
attends over them always, whatever the selection, because that is what it was
pretrained to read. On legal text, dropping them moves Legal-BERT's token
states to a cosine of 0.83 against what it would otherwise produce; with a
frozen backbone nothing can adapt to the difference.

This is what ``attention_mask`` is for, and why it is not ``mask``: one says
what is *read*, the other what may be *chosen*.

.. automodule:: pyhighlights.components.data
   :members:
   :show-inheritance:

Corpora and their loaders have their own page: :doc:`datasets`.
