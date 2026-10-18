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

Mask probes
-----------

A code can also be carried by the shape of a selection, such as how many tokens are kept or where they sit.
:doc:`interlocking` explains how that shape reaches the predictor.
:func:`~pyhighlights.components.codes.mask_probes` asks whether the model's own decisions are recoverable from shape alone.

A class-balanced logistic regression reads the document length, the number of positions kept, the kept share, and a normalized histogram of where the kept positions sit.
It reproduces the model's decisions under five-fold cross-validation.
Each probe is reported against a control in which the decisions are shuffled, and against the gold label.
The channels repeat the probe on the count, the length, the two together, the position histogram, and every feature.
:func:`~pyhighlights.components.codes.kept_by_label` reports how much a model keeps per gold label, which is the simplest code.

The probes need no annotation, and they need ``pip install "pyhighlights[analysis]"``.
They read the document length from the ``tokens`` column.
A run that truncated its inputs should pass the tokens the model actually read.

Selection entropy
-----------------

:func:`~pyhighlights.components.codes.selection_entropy` is the Shannon entropy, in bits, of the tokens a model selects for each class.
It follows Treviso and Martins (2020), *The Explanation Game*, BlackboxNLP, Appendix F.2, where a low entropy marks a trivial protocol.
A code repeats a few tokens per class, so its entropy is low.
The measure itself needs no annotation.

It reports two references.
The first is the entropy of random selections of the same size in the same documents.
The second is the entropy of the annotated tokens, when the corpus has annotations.

Evidence that repeats few tokens has a low entropy too: the toy highlight ``aba`` scores 0.92 bits.
A low entropy therefore reads as a code only when it falls below that of the annotated tokens.
A high entropy does not rule a code out, because a model that signals through selection size can keep arbitrary tokens.

The toy runs of ``pyhighlights-genspp2025`` show both cases.
GenSPP selects for ``abc`` documents at 0.33 bits, against 1.58 bits for the highlight.
Every baseline stays above the highlight's entropy and still uses a size code.

Reading a run
-------------

A run stores the predictions this module reads only when its task sets ``store_predictions``.

.. code-block:: python

   from pyhighlights.components.codes import SelectionCodeAnalyzer

   analyzer = SelectionCodeAnalyzer("results", exclusive=True)
   analyzer.analyze()  # one row per run and seed
   analyzer.codes()  # one row per run, seed and class
   analyzer.probes()  # mask probes, one row per run and seed
   analyzer.kept()  # positions kept per run, seed and gold label
   analyzer.entropy()  # selected-token entropy per run, seed and class

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
