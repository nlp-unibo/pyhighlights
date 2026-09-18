Reference
=========

What a run is made of, one page per part.
Start with :doc:`configurations` if you are looking for a key, and with :doc:`tasks` if you are looking for what a run does with one.
These pages assume a run has already been made, which :doc:`../tutorials/quickstart` covers, and they are what :doc:`../tutorials/custom-model` points at when a method of your own needs one of them.

Running an experiment
---------------------

- :doc:`configurations` is every registration in the library, as searchable cards built from the registry itself.
- :doc:`tasks` is what one experiment is: the stages it runs, and what it writes to disk.
- :doc:`benchmarks` is a grid of tasks, and how a published paper's values are registered beside the library rather than inside it.

Configuring a model
-------------------

- :doc:`embeddings` is what a model starts from, whether a pretrained vector file or one-hot inputs.
- :doc:`metrics` is what a run scores, and what a class-imbalanced corpus needs.
- :doc:`supervision` is training against a highlight annotation rather than against the label alone.
- :doc:`backbones` is the encoder contract and the three implementations that satisfy it.

Reading a run back
------------------

- :doc:`analyzers` turns a results directory into a frame, a table, or the text a model selected.
- :doc:`faithfulness` is sufficiency and comprehensiveness, and what each compares.
- :doc:`diagnostics` records the inside of a forward pass without changing what it computes.
- :doc:`cost` reports what a run spent.
- :doc:`interlocking` is what the bottleneck does and does not guarantee, measured.

Corpora
-------

- :doc:`datasets` is the corpora, their loaders, and the preprocessors that prepare them.
- :doc:`data` is the batch contract every loader and model is written against.

.. toctree::
   :maxdepth: 1
   :hidden:

   configurations
   tasks
   benchmarks
   embeddings
   metrics
   supervision
   backbones
   analyzers
   faithfulness
   diagnostics
   cost
   interlocking
   datasets
   data
