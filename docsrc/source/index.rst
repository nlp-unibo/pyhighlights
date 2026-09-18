pyhighlights
============

Select-then-predict models, backend-independent, for highlight-based explainable AI research.

A select-then-predict model picks a subset of its input and predicts from that subset alone.
The subset, which this library calls the **highlight**, is therefore not a story about the prediction: it is the input to it, so a highlight that omits what mattered produces a worse prediction, which is measurable rather than arguable.

Eight architectures over two backbones, five corpora, and a whole experiment behind one key.

.. code-block:: console

   pip install pyhighlights

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   from cinnamon.registry import Registry
   from pyhighlights.components.analyzers import MetricsAnalyzer
   from pyhighlights.configurations.keys import TOY_TASK

   Registry.build(directory=Path(pyhighlights.__file__).parent)

   Registry.from_key(TOY_TASK, save_path="results", seeds=[0, 1]).run()
   MetricsAnalyzer(directory="results").run()

The corpus, the preprocessing, the model, its metrics and its seeds are all registration keys, so swapping the architecture or the corpus is a value rather than a code change.
What lands on disk is every seed's metrics, a manifest naming the key and the arguments that produced them, and, when asked, the predictions themselves.

.. note::

   The literature calls the selected subset a *rationale* and writes it ``r``, while this library says *highlight* and writes ``h`` throughout, in the loaders, the models, the metrics and the batch fields.
   The two words name the same object, so a paper's ``r`` is this library's ``h``.
   Metric names stay as published, so a reported column still matches the paper it comes from.

Where to start
--------------

- :doc:`concepts/select-then-predict` is the research topic itself: what the architecture is, why a highlight is not an explanation, and the three difficulties every model here inherits.
- :doc:`tutorials/index` are the two walkthroughs: :doc:`tutorials/quickstart` runs one experiment end to end on a synthetic corpus and reads the results back, and :doc:`tutorials/custom-model` writes a method of your own.
- :doc:`models/index` compares the eight implementations and gives each one a page of its own.
- :doc:`project/contributing` is the environment, the checks and the release flow, for a change to the library itself.

.. toctree::
   :maxdepth: 2
   :caption: Start here
   :hidden:

   concepts/select-then-predict
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: Models
   :hidden:

   models/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Project
   :hidden:

   project/contributing

Reference
---------

- :doc:`reference/data` and :doc:`reference/datasets` are the batch contract, and the corpora with the preprocessors that prepare them.
- :doc:`reference/tasks` is what a run is and what it writes, with :doc:`reference/embeddings`, :doc:`reference/metrics` and :doc:`reference/supervision` for what a run is configured with.
- :doc:`reference/analyzers` reads the results back, :doc:`reference/faithfulness` and :doc:`reference/diagnostics` read the model, and :doc:`reference/cost` reports what a run spent.
- :doc:`reference/backbones` is the encoder contract, and :doc:`reference/interlocking` is what the bottleneck does and does not guarantee, measured.
- :doc:`reference/configurations` is every registration in the library, as searchable cards.


What the library is, and is not
-------------------------------

pyhighlights ships tools: models, backbones, loaders, losses, metrics, trainers, and the cinnamon registrations that let a project assemble them into an experiment.

It does not ship an experiment.
Which metrics to log, which aspect of Beer to train on, which sparsity target to aim at, which embedding matrix to freeze, all of that depends on a specific study, so all of it stays a configuration the study writes.
Published numbers live in :doc:`a repository of their own, one per paper <reference/benchmarks>`, so nothing here carries one paper's values.
