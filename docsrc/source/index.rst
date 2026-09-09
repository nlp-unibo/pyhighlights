pyhighlights
============

Select-then-predict models, backend-independent, for highlight-based
explainable AI research.

A select-then-predict model picks a subset of its input and predicts from that
subset alone. The subset — the **highlight** — is therefore not a story about
the prediction, it is the input to it: a highlight that omits what mattered
produces a worse prediction, which is measurable rather than arguable.

Five architectures over two backbones, five corpora, and a whole experiment
behind one key:

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   from cinnamon.registry import Registry
   from pyhighlights.components.analyzers import MetricsAnalyzer
   from pyhighlights.configurations.keys import TOY_TASK

   Registry.build(directory=Path(pyhighlights.__file__).parent)

   Registry.from_key(TOY_TASK, save_path="results", seeds=[0, 1]).run()
   MetricsAnalyzer(directory="results").run()

The corpus, the preprocessing, the model, its metrics and its seeds are all
registration keys, so swapping the architecture or the corpus is a value
rather than a code change. What lands on disk is every seed's metrics, a
manifest naming the key and the arguments that produced them, and — when asked
— the predictions themselves.

.. toctree::
   :maxdepth: 2
   :caption: Start here

   overview
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Reference

   data
   datasets
   models
   tasks
   benchmarks
   configurations

.. toctree::
   :maxdepth: 2
   :caption: Project

   roadmap
   contributing

- :doc:`overview` — what select-then-predict is, how the five architectures
  differ, and what the library deliberately does not ship.
- :doc:`quickstart` — a tutorial that runs: one experiment end to end, then
  reading the results back.
- :doc:`data` and :doc:`datasets` — the batch contract, and the corpora with
  the preprocessors that prepare them.
- :doc:`models` — the backbone contract every algorithm is written against.
- :doc:`tasks` — what a run is, what it writes, and the analyzers that read
  it.
- :doc:`benchmarks` — a published paper's values, registered beside the
  library rather than inside it.
- :doc:`configurations` — the registration idiom, for assembling your own.
