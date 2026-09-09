Data
====

Examples use source-token highlights. ``HighlightCollator`` aligns those labels
to vocabulary tokens or Hugging Face subtokens, then emits model-ready
``InputData`` tensors.

A batch also carries ``word_ids``: which source word each position came from,
``-1`` where none did. No model reads it. It is what turns a selection over
subtokens back into a selection over words once the run is over, which is the
only form a person can read, and it is why
:class:`~pyhighlights.components.analyzers.PredictionAnalyzer` can report
words for a subword model.

.. automodule:: pyhighlights.components.data
   :members:
   :show-inheritance:

Corpora and their loaders have their own page: :doc:`datasets`.
