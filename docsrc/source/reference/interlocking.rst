Interlocking and what the bottleneck guarantees
===============================================

A select-then-predict model claims that the highlight **is** the predictor's input.
Half of that claim is enforced and tested, the other half is open and tested as open, and this page is the measurement of both.
:doc:`../concepts/select-then-predict` states what interlocking is; this page is about what the implementation can and cannot rule out.

The half that holds
-------------------

A word the selector dropped cannot reach the predictor, on every backbone.
``tests/test_mechanics.py`` and ``tests/test_transformer_configurations.py`` check it by changing a dropped word and asserting that the pooled vector the predictor is handed does not move.

The half that does not
----------------------

The **shape** of the mask reaches the predictor even though the dropped words do not.
Two clauses with the same kept words, in the same order, differing only in the gaps between them, produce different predictor inputs.

.. code-block:: text

   backbone                       identical   gap changed   a kept word changed
   GRUBackbone                     0.000000         0.644                 1.989
   TransformerBackbone             0.000000         0.395                 1.803

The numbers are the largest element-wise difference in ``pool(...)``, the single vector the predictor reads, and they are best read as a ratio: changing where the kept words sit moves the predictor's input by about a third of what swapping a kept word for a different word does.
The ``identical`` column is the control.

Two mechanisms, one per family.

* :class:`~pyhighlights.components.models.spp.implementations.GRUBackbone` zeroes a dropped word's embedding but still steps the recurrence over its position, so the number of dropped positions changes the state carried forward.
* :class:`~pyhighlights.components.models.spp.implementations.TransformerBackbone` gives a kept word a different position embedding when the gap before it changes.

Why it matters
--------------

A selector can signal the label through the shape of the mask instead of through the words in it, most cheaply through how many words survive, and a predictor that honestly reads only the highlight can still read that signal.
The highlight is then formally sufficient and semantically empty, which is the private-code failure of :doc:`../concepts/select-then-predict` arriving through the encoder rather than through the vocabulary.
Measured on a legal corpus, a logistic regression given mask shape alone and no text reproduced a model's own decisions at 81.30 unfair F1 against 28.20 for the same features on shuffled decisions, while the highlighted words scored 2.58 on the task itself.

Both cases are marked ``xfail(strict=True)``, so the suite states that the channel is open and will fail loudly on the day it is closed.

Closing it
----------

**Compaction** gathers the kept positions into a shorter sequence rather than zeroing the dropped ones, and it closes both mechanisms at once: a clause of 35 words with 4 kept becomes a sequence of length 4 whatever the gaps were.
Order is preserved, which is what makes the result a highlight rather than a bag of words.

It is off by default and it is not a repair.
Compaction changes what the predictor is trained on, since the predictor already reads a corpus of the selector's construction and compaction makes that corpus shorter and more artificial, so whether the channel it closes was the one that mattered is a measurement rather than a consequence.
One cost is worth knowing before turning it on: a dropped word beyond the batch's widest selection is cut, so the selector gets no gradient telling it to keep that word, which makes exploration weaker than with in-place zeroing.

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_FR

   Registry.from_key(GRU_FR, compact=True)

Diagnosing a run
----------------

Interlocking does not announce itself in the loss, so three readings are what tell a degenerate run from a trained one.

First, the selection metrics, which need no annotation: ``selection_rate``, ``selection_size`` and ``selection_spans`` report what the model actually did.
Second, :class:`~pyhighlights.components.analyzers.HighlightPositionAnalyzer`, which reports where in the document the selection fell, since a model keying on the opening tokens of every document scores like one that found the highlight until you look at where it selected.
Third, the faithfulness diagnostics of :mod:`pyhighlights.components.faithfulness`, where sufficiency compares the prediction from the highlight against the prediction from the full input and comprehensiveness compares the full input against the input with the highlight removed.

The empty-selection repair is worth watching for the same reason.
A sample whose selector marks no valid word would leave the predictor with an empty input, so the highest-scoring word is kept instead, and a repair firing on most of a batch is a selector that has learned nothing while the reported selection rate hides it, since the rescued word counts as a selection like any other.
``pyhighlights.utility.diagnostics`` records the rows it fired on.
