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

Highlight supervision
---------------------

Where a corpus annotates its **training** split, a task can train the selector
against those annotations instead of leaving it to discover them:

.. code-block:: python

   Registry.from_key(TOY_TASK, highlight_supervision=True, highlight_coefficient=0.5)

The flag appends ``HIGHLIGHT_LOSS`` -- masked cross entropy over
``highlight_logits``, ``highlight_true`` and ``mask`` -- to the model the task
names, weighted by ``highlight_coefficient``. Nothing else changes: the same
model key runs either way.

The two settings are different experiments, not two points on one scale. The
unsupervised one is the realistic problem; the supervised one is the ceiling it
is measured against, and its ``val_loss`` carries a term the other has no
equivalent for, so only the metrics compare.

**One annotation guides one head.** A model with several generators supervises
the head its aggregator keeps -- the one every reported metric scores -- and
leaves the rest to diverge, which is what those generators are there for.

**A corpus without training annotations is refused.** Unannotated positions are
padded with ``-1`` and skipped by the criterion, so supervising a corpus
annotated on test alone would train exactly as an unsupervised run does and
report itself as that run's ceiling. The task checks the training split and
raises instead.

GenSPP refuses the flag outright: no gradient reaches its generator, so the
loss would be built and train nothing. Guiding a genetic search means
conditioning the population it draws from, which is open work.

GenSPP
------

GenSPP's generator is not trained: it is searched. A population of generators
is evolved, and a candidate is scored by fitting a predictor on the selections
it makes -- with the generator frozen, so the predictor never teaches the
selector what to select, which is the cooperative equilibrium the other models
have to fight.

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_GENSPP_TASK

   Registry.from_key(TOY_GENSPP_TASK, seeds=[42]).run()

:class:`~pyhighlights.components.tasks.GenSPPTask` is an ``SPPTask`` in every
other respect -- same corpus, preprocessing, metrics, seeds and output files.
Two things differ:

* It names a **search**, not a model. The model key is the search's own; naming
  it twice is a way for the two to disagree about which model was evolved.
* A validation split is required. Fitness is task loss traded against selection
  rate, and both are measured there.

Each candidate's predictor is fitted by a throwaway Lightning trainer, so the
inner training is the same code path every other model trains through --
logging, checkpointing and sanity checks off, since a hundred generations build
one trainer per candidate. Gradients reach the predictor only:
``GenSPP.configure_optimizers`` hands over the predictor's parameters, and the
generator is put back in evaluation mode at the start of every epoch so its
dropout cannot score the same candidate two different ways.

Alongside the usual per-seed files, a GenSPP run writes ``best.ckpt`` -- the
weights the search settled on -- and ``search.json``, the best fitness of every
generation. A search that stopped improving in its tenth generation and one
still climbing when the budget ran out report the same metrics otherwise.

Benchmarks
----------

A paper's table is a grid, not one experiment: every model over every corpus.
:class:`~pyhighlights.components.benchmarks.Benchmark` is that grid — a list of
task keys, run in order, each writing inside the benchmark's own directory.

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_BENCHMARK

   report = Registry.from_key(TOY_BENCHMARK).run()
   report["failed"]   # tasks that raised, by name

A task that raises does not take the rest of the grid with it: the failure is
recorded against that task and the run carries on, since an afternoon of
training should not be lost to one bad configuration. ``strict=True`` turns
that off where a run must be all-or-nothing.

Analyzers
---------

An analyzer reads a results directory back and answers one question about it,
returning a :class:`pandas.DataFrame` rather than printing — so the same
analyzer serves a notebook, a test and a LaTeX table.

:class:`~pyhighlights.components.analyzers.MetricsAnalyzer`
   One row per task, one column per metric, ``mean +/- std`` across seeds. It
   walks every ``results.json`` beneath the directory, so it reads one task or
   a whole benchmark without being told which. A metric a task never measured
   reads as ``-``: a grid rarely reports the same set everywhere, and an
   unannotated corpus has no highlight F1 to give. ``pairs=True`` keeps the
   ``(mean, std)`` tuples, which
   :func:`~pyhighlights.components.analyzers.latex_table` renders as
   ``$12.34_{\pm 0.56}$``, escaping the underscores every metric name
   carries.

:class:`~pyhighlights.components.analyzers.HighlightPositionAnalyzer`
   Where in the document the selector looked, binned as a share of the
   document so lengths are comparable, and how much it kept. A selector that
   has learned nothing still selects something; position is what tells the two
   apart, since a model keying on the opening tokens of every document scores
   like one that found the rationale. A model with several selectors stores one
   mask per head; the analysis reads the head its aggregator keeps, which is
   the one every reported metric scored.

Neither is interactive and neither plots. An analyzer that asks which folder
you meant cannot run unattended, and a figure is a presentation choice that
belongs to whoever is writing the paper.

API
---

.. automodule:: pyhighlights.components.tasks
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.benchmarks
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.analyzers
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.benchmarks
   :members:

.. automodule:: pyhighlights.configurations.tasks
   :members:

.. automodule:: pyhighlights.configurations.metrics
   :members:
