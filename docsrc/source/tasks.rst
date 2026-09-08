Tasks
=====

A task is one experiment, start to finish: a corpus, its preprocessing, a
model, the metrics to score it with, and a list of seeds. It is what a row of a
results table is made of, so reproducing a number means running one key rather
than remembering which loader went with which checkpoint.

.. code-block:: python

   from cinnamon.registry import Registry

   from pyhighlights.configurations.keys import TOY_TASK

   task = Registry.from_key(TOY_TASK, seeds=[42, 1337, 2024])
   results = task.run()
   results["summary"]["test_highlight_f1"]   # {"mean": ..., "std": ..., "values": [...]}

Registered with ``run_method="run"``, so ``cmn-run`` drives the same task from
the command line.

What a run does
---------------

For each seed, in order:

1. ``seed_everything``, then build the model from its key — a fresh one, since
   a select-then-predict model trained through a discrete choice lands
   somewhere different every time, and the spread across seeds is part of the
   result.
2. Train with early stopping on ``monitor`` (``val_loss`` by default) and a
   checkpoint of the best epoch.
3. **Restore that checkpoint before scoring.** Early stopping returns after
   ``patience`` worse epochs, so the weights still in memory are not the ones
   anybody would keep.
4. Score validation and test, and store the test predictions when
   ``store_predictions`` is set.

Then the seeds are summarised — mean, standard deviation and the individual
values for every metric — and written out.

What lands on disk
------------------

.. code-block:: text

   results/<name>/
   ├── results.json      # every seed's metrics, and their summary
   ├── config.json       # the settings that produced them
   ├── seed=42/
   │   ├── epoch=3-step=128.ckpt
   │   └── predictions.pkl
   └── seed=1337/…

Corpus and model
----------------

``loader``, ``preprocessor`` and ``model`` are registration keys, so a task
definition swaps its corpus without touching code. ``preprocessor`` is optional
only where a corpus needs none: HateXplain has no label until an
:class:`~pyhighlights.components.preprocessors.AnnotationAggregator` has run,
and the loader says so rather than guessing one.

Text becomes ids in one of two ways. Name a ``pretrained_model_card`` and the
matching subword tokenizer is used; leave it unset and a vocabulary is fitted
on the **training split alone** — fitting it on evaluation text would leak,
quietly, since nothing downstream can tell where an id came from. Its
``vocabulary_size`` has to match the backbone's ``vocab_size``: an id the
embedding has no row for is a crash at the first batch.

Metrics
-------

Metrics are registered in two layers, as the losses are: a ``torchmetric`` is
the scoring object, and a ``metric`` binds it to the fields it reads. The
binding is what a task names, since the same F1 scores classes or tokens
depending on what it is handed.

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
     - Tokens kept, averaged over samples

Both classification metrics use ``task="multiclass"`` even for a two-class
corpus: a predictor emits one logit per class, and the ``"binary"`` task wants
a single score per sample instead.

Highlight metrics ignore positions annotated with ``-1``, which is what an
unannotated split is padded with — those positions score nothing rather than
counting as negatives. The selection metrics read ``mask`` instead, so they
report what the selector kept whether or not the corpus is annotated at all.

``BINARY_METRICS`` collects the set a two-class corpus wants; Beer, Hotel,
Movies and Toy use it as-is.

API
---

.. automodule:: pyhighlights.components.tasks
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.tasks
   :members:

.. automodule:: pyhighlights.configurations.metrics
   :members:
