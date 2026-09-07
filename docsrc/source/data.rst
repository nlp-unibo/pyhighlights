Data
====

Examples use source-token highlights. ``HighlightCollator`` aligns those labels
to vocabulary tokens or Hugging Face subtokens, then emits model-ready
``InputData`` tensors.

.. automodule:: pyhighlights.components.data
   :members:
   :show-inheritance:

Corpora
-------

Loaders download a corpus once, hand back one ``pandas`` frame per split with
``sample_id``, ``text``, ``tokens``, ``label`` and ``highlights``, and report
how much of each split another split already contains. Beer and Hotel come
from the R2A release of Bao et al., 2018; its only per-token annotation lives
in ``data/target/<task>.train``, which is therefore the default test split.

.. automodule:: pyhighlights.components.datasets
   :members:
   :show-inheritance:
