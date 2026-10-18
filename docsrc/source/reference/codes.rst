Selection codes
===============

A select-then-predict model can classify well from selections that do not contain the evidence.
Its predictor learns from the selector's own selections, so it can decode the label from any selection the selector makes consistently for one class.
On the toy corpus, a lone ``b`` for every ``abc`` document is enough.
The model's accuracy and its own :doc:`faithfulness` scores both look sound, and neither asks whether the evidence was selected.

:mod:`pyhighlights.components.codes` asks the annotation instead, so it needs a corpus with highlight annotations.

Measures
--------

**Complete** means that every annotated position is selected.
With ``exclusive``, a selection is also incomplete when it spells another class's highlight as one contiguous run.
This is meant for a corpus such as the toy one, where each class has one highlight and every document contains fragments of the others.

**Complete in order** keeps the selected tokens in document order and drops their positions.
It is the only view available for a source that stores the selected tokens without their positions.

**Gold recall** is the share of annotated positions that were selected.

**Code** is the most frequent selection of each class, with its length in tokens and the share of the class's documents that selected exactly it.
Its length summed over classes compares runs that settled on codes of different size.

Reading a run
-------------

A run stores the predictions this module reads only when its task sets ``store_predictions``.

.. code-block:: python

   from pyhighlights.components.codes import SelectionCodeAnalyzer

   analyzer = SelectionCodeAnalyzer("results", exclusive=True)
   analyzer.analyze()  # one row per run and seed
   analyzer.codes()  # one row per run, seed and class

:func:`~pyhighlights.components.codes.summarize` and :func:`~pyhighlights.components.codes.codes` accept any frame with the same columns.
The columns are ``run``, ``seed``, ``tokens``, ``selected``, ``highlights``, ``label`` and ``predicted``.
A study that rebuilds selections from checkpoints can therefore use them without stored predictions.

What it does not say
--------------------

The annotation bounds the claim.
On a corpus whose annotators disagree, such as HateXplain, an incomplete selection may still hold evidence that the annotation left out.
A complete selection is also not proof that the predictor read the evidence, since the predictor may decode a code that the selection happens to contain.

API
---

.. automodule:: pyhighlights.components.codes
   :members:
   :show-inheritance:
