Computational cost
==================

Every seed also reports what it took to produce, under ``cost_``.
A table of F1 says which model is better and nothing about what it takes to get there.

``cost_runtime_s``
   The seed, end to end: the model built, trained and scored.
``cost_inference_batch_s``, ``cost_inference_epoch_s``
   The test pass, per batch and whole. Test rather than validation, because it
   is the pass the reported numbers come from and it runs once, on a model
   that has stopped training. The batch figure is the forward passes alone;
   the epoch figure is what a caller waits for, batch loading included.
``cost_memory_mib``
   The high-water mark, in **mebibytes**, what ``nvidia-smi`` and every
   process monitor print. On CUDA it is the run's own, since the counter is
   reset when the seed starts. On CPU it is the **process**'s, which only ever
   rises, a second seed in the same process inherits the first's peak.
``cost_parameters``, ``cost_trainable_parameters``, ``cost_frozen_parameters``
   Every parameter of the scored model, and the two halves of it. A frozen
   encoder is memory and compute at inference however little it learns, and a
   model that freezes most of itself is a different proposition to train than
   one that does not. Counted on the model as it was scored, so a component
   frozen partway through training counts as frozen, and so does a
   generator a genetic search settled on, which descent never moved.
``cost_concurrency``, ``cost_models``
   How many models the seed trained, and how many of them ran at once. One and
   one for a model trained by descent. A genetic search trains its founders
   plus the children of every generation that ran, fewer than
   ``n_generations`` when it reached ``stop_threshold``, scored one per
   worker.
``cost_runtime_per_run_s``
   What **one** model cost, which is what makes the rows comparable:
   ``runtime * concurrency / models``. Wall clock alone would report a search
   as cheap as the hours it happened to take on the machine that ran it.

   There is no per-model *memory* column to go with it. Most of what a run
   holds is the interpreter, torch and the corpus, resident before the first
   candidate exists, so dividing the peak by the workers reports less memory
   per model than a run holds doing nothing.

   ``cost_memory_mib`` is one process's ceiling: the largest of a search's
   workers, which is the figure comparable to a baseline, itself one process.
   A node running the search needs that much again per worker, less whatever
   fork left shared -- ``cost_concurrency`` says how many there were. The
   operating system offers no honest total, since pages shared by fork are
   counted once per process holding them.

:class:`~pyhighlights.components.analyzers.MetricsAnalyzer` reads them like any
other column, so the computational table is the same call with a different prefix::

   MetricsAnalyzer(directory="results", split="cost").analyze()

and the ``test_`` table a paper quotes is unchanged by any of this.

API
---

.. automodule:: pyhighlights.utility.cost
   :members:
