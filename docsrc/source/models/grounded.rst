GroundedSPP: select-then-predict over a knowledge base
======================================================

``GroundedSPP`` is a select-then-predict model for corpora that explain their labels with free text rather than with spans.
Its motivating corpus is ToS-100, from Ruggeri, Lagioia, Lippi and Torroni, 2022, *Detecting and explaining unfairness in consumer contracts through memory networks*, Artificial Intelligence and Law 30(1), 59-92, https://doi.org/10.1007/s10506-021-09288-2.
That corpus records which legal rationales make a clause unfair, and it does not record which words carry them.

Problem
-------

A span annotation says which words of an input explain its label.
A knowledge base annotation says which entries of a fixed set of texts explain it.
The entries form the knowledge base ``K = {k_1, ..., k_M}``, which is a property of the corpus and is shared by every example.
A model grounded in such a base has to answer which entries an input instantiates, and it has to keep the select-then-predict guarantee: the predictor reads only the highlight.

Method
------

For every pair of an input ``x`` and an entry ``k_i``, the model extracts a highlight pair.
The input-side highlight ``h_i`` holds the words of ``x`` that match ``k_i``.
The entry-side highlight holds the words of ``k_i`` that match ``x``.

One selector serves both sides.
It reads the input's word states concatenated with the pooled state of the entry, and the entry's word states concatenated with the pooled state of the input.
The selector is therefore built at twice the backbone's output width.

A comparer scores each pair from the two pooled highlights.
The registered comparer, ``EntailmentComparer``, is an MLP over ``[u; v; u - v; u * v]``.
The ``u - v`` term makes the score directed, because an input instantiates an entry rather than resembling it.
A hard Gumbel over the comparer's two logits gives the gate, which names the subset ``K_x`` of entries the input instantiates.

The predictor reads the union of the input-side highlights over all entries.
The union is not gated by ``K_x``.
A gated union fails on the majority class.
An example that gates every entry off leaves an empty union, and the empty-selection repair keeps one word of it.
The predictor then learns that a one-word highlight means the negative class.
Each ``h_i`` is still conditioned on its own entry, so the selection depends on the knowledge base even though the gate does not reach it.

Training
--------

One optimizer takes one step per batch.
The batch and the knowledge base are encoded once each per step, so the encoder cost does not grow with ``M``.
The conditioned states are materialised as a ``[B, M, T, 2D]`` tensor, which takes ``B * M * T * 2D * 4`` bytes in float32.

The registered configurations carry four losses.
The classification loss and the two readability penalties are the ones every select-then-predict model carries, applied to the union.
The knowledge loss scores the comparer's logits against the gold links of each example.
A knowledge sparsity loss is registered and left out of the default list, because its target depends on the corpus.

Faithfulness
------------

The token-level sufficiency and comprehensiveness of :mod:`pyhighlights.components.faithfulness` apply unchanged.
The knowledge axis adds two terms, written with ``K`` for the whole base and ``K_x`` for the entries the model named:

.. code-block:: text

   rationale sufficiency       = p(y_hat | x, K) - p(y_hat | x, K_x)
   rationale comprehensiveness = p(y_hat | x, K) - p(y_hat | x, K \ K_x)

Both follow the token-level convention of reference minus restricted.
Lower rationale sufficiency is better, and higher rationale comprehensiveness is better.
Since the union is ungated, restricting the base restricts which pairs enter the union.

Implementation
--------------

The corpus loader provides the knowledge base through ``HighlightLoader.knowledge``, with the entries in the order the annotation indexes them.
The task tokenizes the base once with the corpus tokenizer and hands it to ``GroundedSPP.load_knowledge``.
The base is neither a buffer nor a parameter, so a checkpoint does not carry a copy of the corpus.
It moves to the module's device the first time the model reads it.

``GroundedSPP.forward`` returns a ``GroundedSPPOutput``.
It carries every field of an ``SPPOutput`` and adds the comparer's logits, the gate, the per-entry score, and the input-side and entry-side highlight of every pair.

Configuration
-------------

Two keys, one per backbone.

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_GROUNDED, TRANSFORMER_GROUNDED

A corpus without a knowledge base cannot run these keys, since ``GroundedSPP.knowledge`` refuses to read a base that was never loaded.
A link metric has to be registered with the number of entries in the base, because ``torchmetrics`` needs that number when the metric is built.
