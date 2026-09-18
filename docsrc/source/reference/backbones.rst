Backbones
=========

A backbone is an encoder, and it is the one piece of a model this library does not write an algorithm against.
An architecture asks a backbone for three things, which are ``encode``, ``pool`` and ``output_size``, and nothing in FR, MGR, MCD, DR, DAR, MRD, G-RAT or GenSPP mentions a GRU or a Transformer anywhere.
Running an architecture on a different encoder is therefore a registration key rather than a code change.

The contract
------------

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Member
     - What it returns
   * - ``encode(features, mask, selection_mask=None)``
     - Token states shaped ``[B, T, D]``. ``selection_mask`` is how a dropped word is kept out of the predictor's input.
   * - ``pool(states, mask)``
     - One vector per sequence, shaped ``[B, D]``, which is what a predictor or a selector head reads.
   * - ``output_size``
     - The width of both, so a head can be built before a batch exists.
   * - ``load_embeddings(matrix)``
     - Optional. A backbone whose tokens are already embedded by something else says so rather than ignoring the tensor.

What ships
----------

Three implementations, and each answers the discrete-choice problem in the same way while reading text differently.

:class:`~pyhighlights.components.models.spp.implementations.GRUBackbone` embeds tokens from a table and encodes them with a GRU, packing the batch so padding stays out of the recurrence.
It is the architecture every published select-then-predict implementation uses, over a frozen GloVe table.

:class:`~pyhighlights.components.models.spp.implementations.TransformerBackbone` encodes with a pretrained transformer read through ``transformers``, which is an optional dependency, and ``freeze_transformer`` decides whether its weights move.
Selection happens over words rather than subtokens, so a word's subtoken states are folded into one state after the encoder and never before it, which keeps the encoder on the distribution it was pretrained on.

:class:`~pyhighlights.components.models.spp.implementations.StackedBackbone` puts the two together: a frozen pretrained encoder underneath, and a GRU trained from scratch on top.
It is the shape the papers use with a better frozen representation in place of GloVe, and it avoids two failure cases at once.
A frozen transformer read by a linear selector trains a few thousand parameters, which is a probe rather than any of these architectures, while fine-tuning the transformer makes one learning rate wrong for the model, since ``1e-3`` destroys a pretrained encoder and ``2e-5`` barely moves a selector initialised from scratch.

Where a study does fine-tune a pretrained encoder, ``encoder_lr`` is the field for it: the parameters inside a backbone train at that rate and everything above them at the optimizer's own.
A rate is defined by where a parameter sits rather than by whether it arrived pretrained, so a backbone is the encoder and a selector, predictor or guider head is not.

Keys
----

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Key
     - What it builds
   * - ``GRU_BACKBONE``
     - A bidirectional GRU over an embedding table.
   * - ``TRANSFORMER_BACKBONE``
     - A pretrained transformer, fine-tuned.
   * - ``FROZEN_TRANSFORMER_BACKBONE``
     - The same transformer with its weights held.
   * - ``STACKED_BACKBONE``
     - A frozen transformer read by a GRU.
   * - ``GENSPP_GRU_BACKBONE``, ``GENSPP_TRANSFORMER_BACKBONE``
     - The two above with their embeddings frozen, which is what a genetic search over generator parameters needs.

Pooling is the backbone's own operation and models never reimplement it, since a recurrent encoder pools by maximum and a transformer by masked mean, and a reimplementation would hand the comparer a different summary than the predictor sees.
A row with nothing unmasked pools to zeros rather than to negative infinity, which is an honest reading of an empty complement rather than a number to propagate.

What the bottleneck does and does not guarantee is measured on its own page: :doc:`interlocking`.

API
---

.. automodule:: pyhighlights.components.models.spp.base
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.models.spp.implementations
   :members:

.. automodule:: pyhighlights.configurations.backbones
   :members:
