Quickstart
==========

A tutorial that runs. Every block below is real, in order, against the
synthetic ``toy`` corpus so it needs no download and finishes in seconds.

.. code-block:: console

   pip install pyhighlights

Transformer backbones are an extra, because a GRU run should not pull in
``transformers``:

.. code-block:: console

   pip install "pyhighlights[transformers]"

1. Build the registry
---------------------

Nothing is registered until something asks. One call executes every
registration the library ships and reports which keys survived validation:

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   from cinnamon.registry import Registry

   valid, invalid = Registry.build(directory=Path(pyhighlights.__file__).parent)

``invalid`` is empty for the library itself. A key lands there when its
configuration fails validation — a grid that varies the embedding source into
an impossible combination, say — and it is dropped before anything trains
rather than raising halfway through the sweep that reaches it.

2. Run an experiment
--------------------

A task is one experiment start to finish: a corpus, its preprocessing, a
model, its metrics, and a list of seeds. It is a single key, and the pieces it
names are keys too, so swapping the corpus or the model is a value rather than
a code change.

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_TASK

   task = Registry.from_key(
       TOY_TASK,
       save_path="results",
       seeds=[0, 1],
       store_predictions=True,
       trainer_args={"accelerator": "cpu", "max_epochs": 1},
   )
   results = task.run()

Each seed trains from scratch, restores the checkpoint that scored best on
validation, and is evaluated on validation and test. ``results["summary"]``
holds the mean and standard deviation of every metric across seeds:

.. code-block:: python

   >>> sorted(results["summary"])
   ['test_accuracy', 'test_classification', 'test_contiguity', 'test_f1',
    'test_highlight_f1', 'test_highlight_iou', 'test_loss',
    'test_selection_rate', 'test_selection_size', 'test_sparsity',
    'val_accuracy', ...]

Seeds are a list rather than a number on purpose. One run of a
select-then-predict model says very little: the selector is trained through a
discrete choice, and the spread across seeds is part of the result.

.. note::
   The numbers this produces are meaningless. ``toy`` is a synthetic corpus
   and this is one epoch on it — the point is that the shape is right, not
   that the accuracy is good.

3. Read what landed on disk
---------------------------

.. code-block:: text

   results/toy/2026-09-09T17-55-22/
   ├── results.json               # every seed's metrics, and their summary
   ├── manifest.json              # what the run was
   ├── predictions-seed=0.pkl
   ├── predictions-seed=1.pkl
   ├── seed=0/
   │   └── epoch=0-step=8.ckpt
   └── seed=1/…

The directory is stamped with the moment the run began, and a second run never
overwrites the first: two runs of one task are two results to compare.

``manifest.json`` is what makes the directory worth keeping:

.. code-block:: json

   {
     "started": "2026-09-09T17-55-22",
     "component": "pyhighlights.components.tasks.SPPTask",
     "key": "name=task--tags=['toy']--namespace=pyhighlights",
     "build_args": {"seeds": [0, 1], "store_predictions": true,
                    "trainer_args": {"accelerator": "cpu", "max_epochs": 1}},
     "versions": {"python": "3.13.15", "pyhighlights": "0.2.0",
                  "cinnamon-core": "2.0.3", "torch": "2.14.0",
                  "lightning": "2.6.5"},
     "settings": {"model": {"key": "name=model--tags=['fr', 'gru']--namespace=pyhighlights",
                            "selector_backbones": {"hidden_size": 128,
                                                   "bidirectional": true},
                            "predictor": {"num_classes": 2}}}
   }

The key and the build args are the run, replayable. The ``settings`` tree is
every key resolved into the numbers behind it, recursively, so the file states
the hidden size and the learning rate rather than the name of the place they
came from. The versions are there because a metric that moved between two runs
of the same configuration is a version difference or nothing at all.

:doc:`tasks` has the whole layout.

4. Read the metrics back
------------------------

An analyzer reads a results directory and returns a
:class:`pandas.DataFrame`, so the same analyzer serves a notebook, a test and
a LaTeX table.

.. code-block:: python

   from pyhighlights.components.analyzers import MetricsAnalyzer

   MetricsAnalyzer(directory="results", metrics=["accuracy", "highlight_f1"]).run()

.. code-block:: text

   task                 run  seeds          accuracy      highlight_f1
    toy 2026-09-09T17-55-22      2 0.4062 +/- 0.0312 0.0000 +/- 0.0000

It walks every ``results.json`` beneath the directory, so it reads one task or
a whole benchmark without being told which. A metric a task never measured
reads as ``-`` rather than as zero. ``pairs=True`` keeps the ``(mean, std)``
tuples, which :func:`~pyhighlights.components.analyzers.latex_table` typesets
as ``$12.34_{\pm 0.56}$``.

5. Read what the model selected
-------------------------------

A stored prediction is token ids and masks — enough to score, unreadable on
its own. :class:`~pyhighlights.components.analyzers.PredictionAnalyzer` joins
it back to the corpus it came from:

.. code-block:: python

   from pyhighlights.components.analyzers import PredictionAnalyzer

   frame = PredictionAnalyzer(directory="results").analyze()
   frame[["seed", "sample_id", "label", "predicted", "rationale"]].head(3)

.. code-block:: text

    seed  sample_id  label  predicted rationale
       0          0      0          1        w6
       0          1      1          1       w18
       0          2      0          1        w6

The run's ``manifest.json`` names the loader and the preprocessor, so the
analyzer rebuilds exactly the split the run trained against and joins on
``sample_id``. Selections are folded from token positions back to words, so a
subword model reports words like every other one.

The companion question is *where* it selected:

.. code-block:: python

   from pyhighlights.components.analyzers import HighlightPositionAnalyzer

   HighlightPositionAnalyzer(directory="results", bins=4).run()

A selector that has learned nothing still selects something. Position is what
tells the two apart: a model keying on the opening tokens of every document
scores like one that found the rationale, until you look at where it selected.

6. Swap the corpus, swap the model
----------------------------------

Both are keys, so both are arguments:

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_MGR, MOVIES, TRANSFORMER_FR

   Registry.from_key(TOY_TASK, model=GRU_MGR)                 # a different architecture
   Registry.from_key(TOY_TASK, model=TRANSFORMER_FR)          # a different backbone
   Registry.from_key(TOY_TASK, loader=MOVIES)                 # a different corpus

An override is recorded in the manifest alongside the key, so a run launched
this way is still replayable.

Registering the combination instead of overriding it is what a study does:
:doc:`configurations` has the idiom, and :doc:`benchmarks` shows a published
paper's values registered in a namespace of their own.

7. Run a grid
-------------

A benchmark is a list of task keys, run in sequence, each writing inside the
benchmark's directory:

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_BENCHMARK

   Registry.from_key(TOY_BENCHMARK, save_path="results").run()

One task failing does not take the grid with it — a night of training should
not be lost to one bad configuration — and ``strict=True`` turns that off
where a run must be all-or-nothing. Point ``MetricsAnalyzer`` at the
benchmark's directory afterwards and every task is one row.

From the command line
---------------------

Every registration above carries a ``run_method``, so cinnamon's CLI offers
the same runs without a script. ``cmn-build`` reports what registers and what
does not:

.. code-block:: console

   cmn-build --directory path/to/your/project

``cmn-run`` lists the runnable keys and runs the one you pick. It prompts, so
it needs cinnamon's CLI extra:

.. code-block:: console

   pip install "cinnamon-core[cli]"
   cmn-run
