Token vectors
=============

A model starts from token vectors, and a task says where they come from in one of three ways, which are mutually exclusive.
Name a ``pretrained_model_card`` and the tokens are embedded by that pretrained encoder, which is what :doc:`backbones` covers.
Name ``embeddings`` and a vector file is fitted against the training split, which is the rest of this page.
Name neither and set ``one_hot=True`` and the tokens carry no pretrained representation at all.

Pretrained token vectors
------------------------

``embeddings`` names a GloVe-style vector file, one line per token, the token and then its vector, and it is what a reproduction of the published architectures usually needs.

.. code-block:: python

   task = SPPTask(
       loader=HATEXPLAIN,
       model=GRU_FR,
       embeddings="glove.twitter.27B.25d.txt",
   )

The file's width has to be the backbone's ``embedding_dim``.
The ids the vocabulary hands out start at ``2``: rows ``0`` and ``1`` are the padding and unknown ids, both zero, kept apart so that a highlight over an unknown word is not a highlight over padding.
Its **length** does not have to be anything: the matrix is handed to the model through
:meth:`~pyhighlights.components.models.spp.base.SPP.load_embeddings`, which
sizes the table to it, so ``vocab_size`` is not a number the configuration has to have guessed.
Whether the table then trains is still the backbone's ``freeze_embeddings``, untouched by the load.

``vocabulary_from`` decides which tokens are read, and the two answers are different experiments rather than a tidiness choice.

``"corpus"``, the default, reads only the tokens the **training split** uses.
A token the training split never saw is unknown at evaluation whatever the file covers, and ``pretrained_tokens_only`` decides what happens to a training token the file has no vector for, dropped by default, so every row is a released vector, since a random row inside a frozen table is noise nothing can learn away.
Set it to ``False`` to keep the token with a random row instead.

``"vectors"`` takes the file's own vocabulary whole.
Nothing the file covers is ever unknown.
This is not a leak: the file is external, and which words it holds says nothing about which split uses them, which is why the choice is about fidelity rather than hygiene.
The released GenSPP baselines embed this way (``use_pretrained_only=True``), and on their HateXplain splits the difference is 5.4% of validation tokens and 5.6% of test tokens, embedded as zero under ``"corpus"`` and as their GloVe vector under ``"vectors"``.

``"vectors"`` needs ``embeddings`` and refuses to build without it.
``requires_embeddings=True`` asks for the same refusal while keeping the corpus vocabulary, for a reproduction whose numbers are a vector file's even though its vocabulary is not.

``embeddings`` and ``pretrained_model_card`` are mutually exclusive: a subword tokenizer brings its own embeddings.

One-hot inputs
--------------

A corpus whose tokens are symbols rather than words has nothing to pretrain and nothing to learn:
``one_hot_embeddings=<width>`` builds the table instead of reading one, so every token is orthonormal to every other and the padding and unknown ids, rows ``0`` and ``1``, are zero.
A frozen *random* table is not the same corpus to learn from, its rows are neither unit-length nor orthogonal, so the symbols arrive entangled.
The width has to match the backbone's ``embedding_dim``, and may exceed the vocabulary, which leaves columns that are always zero.

It is one of three ways a task embeds its tokens, and they are mutually exclusive: a pretrained model card, a vector file, or this.

API
---

.. automodule:: pyhighlights.utility.embeddings
   :members:
