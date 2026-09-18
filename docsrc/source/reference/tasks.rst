Tasks
=====

A task is one experiment, start to finish: a corpus, its preprocessing, a model, the metrics to score it with, and a list of seeds.
It is what a row of a results table is made of, so reproducing a number means running one key rather than remembering which loader went with which checkpoint.

.. code-block:: python

   from cinnamon.registry import Registry

   from pyhighlights.configurations.keys import TOY_TASK

   task = Registry.from_key(TOY_TASK, seeds=[42, 1337, 2024])
   results = task.run()
   results["summary"]["test_highlight_f1"]   # {"mean": ..., "std": ..., "values": [...]}

Registered with ``run_method="run"``, so ``cmn-run`` drives the same task from the command line.

Anything decided before a run is a parameter of
:class:`~pyhighlights.configurations.tasks.TaskConfig`: what monitors the run, which vector file to read, whether predictions and faithfulness are reported.
A key therefore records what was asked for rather than the part of it somebody remembered to register.
The kwargs above override a key at build time; they are not the only way to set a value.
A value a run *computes* cannot be a parameter at all: the embedding matrix is fitted against the training split and reaches the model as a tensor, and a vocabulary size measured by a tokenizer is known only once that split has been read.

Constraints between parameters are declared as cinnamon conditions rather than checked in the component, so an invalid combination is rejected while the registry expands keys.
A task that names both ``pretrained_model_card`` and ``embeddings`` fails the ``one_embedding_source`` condition, and a grid varying the embedding source drops that combination before anything trains.

What a run does
---------------

For each seed, in order:

1. ``seed_everything``, then build a fresh model from its key, since
   a select-then-predict model trained through a discrete choice lands
   somewhere different every time, and the spread across seeds is part of the
   result.
2. Train under the ``callbacks`` the task names, early stopping and a
   checkpoint of the best epoch, both on ``val_loss`` by default. **They have
   to monitor the same quantity**, and a task that is given two refuses to
   build rather than reporting a model its own stopping rule did not choose;
   :class:`~pyhighlights.components.callbacks.GeneralizationLossScore` is how
   two quantities become the one they can agree on.
3. **Restore that checkpoint before scoring.** Early stopping returns after
   ``patience`` worse epochs, so the weights still in memory are not the ones
   anybody would keep.
4. Score validation and test, and store the test predictions when
   ``store_predictions`` is set, one ``predictions-seed=<seed>.pkl`` per seed,
   in the run directory.

Then the seeds are summarised and written out, as the mean, the standard deviation and the individual values of every metric.

What lands on disk
------------------

.. code-block:: text

   results/<name>/<started>/
   ├── results.json                 # every seed's metrics and costs, and their summary
   ├── manifest.json                # the whole configuration tree, and the versions
   ├── predictions-seed=42.pkl      # when ``store_predictions`` is set
   ├── predictions-seed=1337.pkl
   ├── seed=42/
   │   └── epoch=3-step=128.ckpt
   └── seed=1337/…

The weights are the one part of that tree nothing downstream reads: the task restores the best checkpoint itself before scoring, and an analyzer reads ``results.json`` and the stored predictions.
On a grid of fine-tuned transformer cells they are also most of the bytes, since a Legal-BERT MGR cell is over a gigabyte per seed.
``keep_checkpoints=False`` writes the checkpoint, restores it, scores, and then deletes it; ``save_weights_only`` keeps it but drops the optimizer state, which is only needed to resume training and no task resumes.
Both are off by default, and the trade the first one makes is that a number cannot be re-scored without training again.

``<started>`` is the moment the run began, ``2026-09-09T16-13-00``.
A run never overwrites an earlier one: two runs of the same task are two results to compare, and the second quietly replacing the first is a measurement lost to a re-run somebody forgot they had already done.
Two runs inside one second get ``…-2`` appended rather than sharing a directory.

``manifest.json`` is what makes the directory worth keeping.
A task's own attributes are not enough, since a task holds *keys*: recording them writes ``name=model--tags=['fr','gru']`` and leaves the hidden size, the sparsity threshold and the learning rate behind that key nowhere in the record.
:func:`~pyhighlights.utility.manifest.describe` replaces every key with the
configuration it names, recursively, so the file states the numbers the run used:

.. code-block:: json

   {
     "started": "2026-09-09T16-13-00",
     "component": "pyhighlights.components.tasks.SPPTask",
     "key": "name=task--tags=['fr', 'gru', 'movies']--namespace=pyhighlights",
     "build_args": {"seeds": [0, 1, 2]},
     "versions": {"python": "3.13.15", "pyhighlights": "0.2.0",
                  "cinnamon-core": "2.0.3", "torch": "2.14.0",
                  "lightning": "2.6.5"},
     "settings": {
       "callbacks": [{"@key": "name=callback--tags=['early_stopping','loss']--namespace=pyhighlights",
                      "monitor": "val_loss", "patience": 5}],
       "model": {
         "@key": "name=model--tags=['fr', 'gru']--namespace=pyhighlights",
         "selector_backbones": {"hidden_size": 128, "bidirectional": true},
         "losses": [{"name": "sparsity", "loss": {"threshold": 0.15}}],
         "optimizer": {"lr": 0.001}
       }
     }
   }

Predictions sit beside the run rather than inside a checkpoint directory.
They belong to the run: a reader that finds them next to a checkpoint can say which seed produced them and not which run, and the analyzers report both.

The versions are there because a metric that moved between two runs of the same configuration is a version difference or nothing at all.
Private attributes are absent: they are what the run *built*, the embedding matrix fitted against the training split among them, and no more a setting than the trained weights are.

``key`` and ``build_args`` are what make the file replayable rather than merely readable.
The key alone rebuilds the *registered defaults*, not the run that was launched, so the overrides go down beside it: a manifest saying ``seeds: [0, 1, 2]`` is a run somebody can re-run, and one saying only the key is not. cinnamon annotates every component it builds with both, so a task constructed directly rather than through a key reports ``null`` for each.

Build args are resolved like everything else, so an override that names a key, as in ``Registry.from_key(TASK, model=GRU_MGR)``, is written out as the settings behind it rather than as a name.

Corpus and model
----------------

``loader``, ``preprocessor`` and ``model`` are registration keys, so a task definition swaps its corpus without touching code.
``preprocessor`` is optional only where a corpus needs none:
HateXplain has no label until an
:class:`~pyhighlights.components.preprocessors.AnnotationAggregator` has run,
and the loader says so rather than guessing one.

Text becomes ids in one of two ways.
Name a ``pretrained_model_card`` and the matching subword tokenizer is used; leave it unset and a vocabulary is fitted on the **training split alone**, since fitting it on evaluation text would leak quietly and nothing downstream can tell where an id came from.
Its ``vocabulary_size`` has to match the backbone's ``vocab_size``: an id the embedding has no row for is a crash at the first batch.

Everything else a run configures has a page of its own:
:doc:`embeddings` for the token vectors a model starts from, :doc:`metrics` for what is scored, :doc:`supervision` for training against a highlight annotation, :doc:`faithfulness` for the sufficiency and comprehensiveness diagnostics, :doc:`diagnostics` for reading the inside of a forward pass, :doc:`cost` for what a run spent, :doc:`analyzers` for reading the results back, and :doc:`benchmarks` for running a grid of tasks.

API
---

.. automodule:: pyhighlights.components.tasks
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.callbacks
   :members:

.. automodule:: pyhighlights.utility.manifest
   :members:

.. automodule:: pyhighlights.configurations.tasks
   :members:

.. automodule:: pyhighlights.configurations.callbacks
   :members:
