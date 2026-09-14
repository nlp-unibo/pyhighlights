Models
======

Algorithms depend on ``SPPBackbone`` rather than GRU or Transformer details.
Output tensors retain a stable head dimension.

Base contracts
--------------

.. automodule:: pyhighlights.components.models.spp.base
   :members:
   :show-inheritance:

Algorithms
----------

.. automodule:: pyhighlights.components.models.spp.fr
   :members:

.. automodule:: pyhighlights.components.models.spp.genspp
   :members:

.. automodule:: pyhighlights.components.models.spp.mgr
   :members:

.. automodule:: pyhighlights.components.models.spp.mcd
   :members:

.. automodule:: pyhighlights.components.models.spp.grat
   :members:

.. automodule:: pyhighlights.components.models.spp.dr
   :members:

.. automodule:: pyhighlights.components.models.spp.mrd
   :members:

.. automodule:: pyhighlights.components.models.spp.dar
   :members:

Grounded in a knowledge base
----------------------------

A corpus may explain its labels in free text rather than in spans. Where it
does, a grounded model extracts a highlight pair for every knowledge base
entry -- the words of the input matching the entry, and the words of the entry
matching the input -- and names the subset the input instantiates.

.. automodule:: pyhighlights.components.models.spp.grounded
   :members:

Backends
--------

.. automodule:: pyhighlights.components.models.spp.implementations
   :members:

What the bottleneck does and does not guarantee
-----------------------------------------------

A select-then-predict model claims that the highlight **is** the predictor's
input. The library enforces the half of that claim about *words*: a word the
selector dropped cannot reach the predictor, on every backbone, and
``tests/test_mechanics.py`` and ``tests/test_transformer_configurations.py``
check it by changing a dropped word and asserting the pooled vector does not
move.

**The other half does not hold yet.** The *shape* of the mask reaches the
predictor even though the dropped words do not. Two clauses with the same kept
words, in the same order, differing only in the gaps between them, produce
different predictor inputs:

.. code-block:: text

   backbone                       identical   gap changed   a kept word changed
   GRUBackbone                     0.000000         0.644                 1.989
   TransformerBackbone             0.000000         0.395                 1.803

The numbers are the largest element-wise difference in ``pool(...)``, the
single vector ``predictor`` is handed. Read them as a ratio: changing where the
kept words sit moves the predictor's input by about a **third** of what
swapping a kept word for a different word does. The ``identical`` column is the
control.

Two mechanisms, one per family:

* :class:`~pyhighlights.components.models.spp.implementations.GRUBackbone`
  zeroes a dropped word's embedding but still steps the recurrence over its
  position, so the *number* of dropped positions changes the state carried
  forward.
* :class:`~pyhighlights.components.models.spp.implementations.TransformerBackbone`
  gives a kept word a different position embedding when the gap before it
  changes.

**Why it matters.** A selector can signal the label through the shape of the
mask instead of through the words in it — most cheaply through *how many* words
survive — and a predictor that honestly reads only the highlight can still read
that signal. The highlight is then formally sufficient and semantically empty.
Measured on a legal corpus, a logistic regression given mask shape alone and no
text reproduced a model's own decisions at 81.30 unfair F1 against 28.20 for
the same features on shuffled decisions, while the highlighted words scored
2.58 on the task itself.

Both cases are marked ``xfail(strict=True)``, so the suite says the channel is
open and will fail loudly when it is closed. **Compaction** — gathering the
kept positions into a shorter sequence rather than zeroing the dropped ones —
closes both mechanisms at once, and is a change to what a backbone does rather
than a repair, which is why it is not applied silently.
