Metrics
=======

What a run scores, and the one field a class-imbalanced corpus cannot be read without.

Metrics are registered in two layers, as the losses are: a ``torchmetric`` is the scoring object, and a ``metric`` binds it to the fields it reads.
The binding is what a task names, since the same F1 scores classes or tokens depending on what it is handed.

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Key
     - Reads
     - Reports
   * - ``ACCURACY_METRIC``
     - ``class_logits``, ``y_true``
     - Classification accuracy, two classes
   * - ``F1_METRIC``
     - ``class_logits``, ``y_true``
     - Macro F1, two classes
   * - ``MULTICLASS_ACCURACY_METRIC``
     - ``class_logits``, ``y_true``
     - Accuracy over three classes, for HateXplain
   * - ``MULTICLASS_F1_METRIC``
     - ``class_logits``, ``y_true``
     - Macro F1 over three classes
   * - ``HIGHLIGHT_F1_METRIC``
     - ``highlight_mask``, ``highlight_true``
     - Token F1 against the annotation
   * - ``HIGHLIGHT_IOU_METRIC``
     - ``highlight_mask``, ``highlight_true``
     - Token intersection over union
   * - ``SELECTION_RATE_METRIC``
     - ``highlight_mask``, ``mask``
     - Share of the document the selector kept
   * - ``SELECTION_SIZE_METRIC``
     - ``highlight_mask``, ``mask``
     - Words kept, averaged over samples
   * - ``SELECTION_SPANS_METRIC``
     - ``highlight_mask``, ``mask``
     - How many contiguous spans the selection falls into
   * - ``CLASS_F1_METRIC``
     - ``class_logits``, ``y_true``
     - F1 of one class rather than an average over all of them
   * - ``HIGHLIGHT_PRECISION_METRIC``, ``HIGHLIGHT_RECALL_METRIC``
     - ``highlight_mask``, ``highlight_true``
     - The two halves of the highlight F1, which a study reports when a selector is precise and short or broad and complete

Both classification metrics use ``task="multiclass"`` even for a two-class corpus: a predictor emits one logit per class, and the ``"binary"`` task wants a single score per sample instead.

Highlight metrics ignore positions annotated with ``-1``, which is what an unannotated split is padded with, those positions score nothing rather than counting as negatives.
The selection metrics read ``mask`` instead, so they report what the selector kept whether or not the corpus is annotated at all.

``BINARY_METRICS`` collects the set a two-class corpus wants, and Beer, Hotel, Movies and Toy use it as it stands.

``CLASS_F1_METRIC`` is the one to reach for where a corpus is imbalanced enough that a macro average hides the answer: a corpus in which one class is 99.5% of the rows is scored at roughly 0.5 by a model that never predicts the other one.
The selection metrics need no annotation at all, which is what makes them the first thing to read on a corpus that ships none, and :doc:`interlocking` is what they are read for.

Class weights
-------------

Where a corpus is class-imbalanced, the loss needs one number per class, and where that number came from decides whether the run can be repeated.
:class:`~pyhighlights.components.tasks.ClassWeightsTask` is a run whose whole
result is those numbers:

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_CLASS_WEIGHTS_TASK

   Registry.from_key(TOY_CLASS_WEIGHTS_TASK, save_path="results").run()

.. code-block:: json

   {
     "split": "train",
     "weights": [1.0, 1.0],
     "counts": {"0": 32, "1": 32},
     "rows": {"train": 64, "val": 16, "test": 16},
     "labels": {"train": {"0": 32, "1": 32}, "val": {"0": 8, "1": 8},
                "test": {"0": 8, "1": 8}}
   }

It trains nothing and takes no seeds.
What it writes is the usual ``results.json`` and ``manifest.json``, so the weights arrive with the key of the corpus and of the preprocessing that produced them, which is what makes them worth copying into a
:class:`~pyhighlights.configurations.losses.CrossEntropyConfig`, where every
training run's manifest then records them.

The reading itself is
:class:`~pyhighlights.components.preprocessors.ClassWeights`, a preprocessor
that changes no row.
Being a step of the pipeline is the point: it runs over the split the study trains on, *after* whatever filtering and aggregation came before it, since that is what changes the frequencies.

Declaring the numbers rather than computing them at training time is deliberate.
A fixed split has fixed frequencies, and a declared weight is in the manifest of every run that used it, where one computed inside training exists only for the length of the process.

API
---

.. automodule:: pyhighlights.utility.metrics
   :members:

.. automodule:: pyhighlights.configurations.metrics
   :members:
