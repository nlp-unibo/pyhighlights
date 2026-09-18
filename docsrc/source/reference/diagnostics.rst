Diagnostics
===========

A run either finishes or raises, and nothing in between is visible.
When a number comes out wrong, ``diagnostics=True`` writes what each stage of the pipeline actually held into ``diagnostics.log``, beside that run's ``results.json``:

.. code-block:: python

   task = Registry.from_key(
       TOY_TASK,
       diagnostics=True,
       trainer_args={"fast_dev_run": True},
   )
   task.run()

Seven stages report, in the order a batch meets them: the frames the loader parsed, each step of a :class:`~pyhighlights.components.preprocessors.Pipeline`, the batch the collator assembled with its two axes side by side, the states a backbone produced, the selection before and after the empty-selection repair, the mask the predictor actually reads, the namespace every loss binds to with each term's value, and, once per split since neither changes between batches, the namespace the metrics bind to with the fields each of them names.
A tensor reports its shape, dtype, device, non-finite count and range, and a mask, which is a batch and one axis of nothing but zeros and ones, reports how many of them are on.
That last count is where a mask which is neither zero nor one, a word dropped on the word axis but still read on the subtoken axis, or a ``nan`` inside a pooled state shows up, and where a metric shows nothing.

**Only under a smoke test.** The record is per batch, so a full run writes gigabytes of it and pays the formatting on every step.
A task asked to diagnose a run whose *training* batches nobody bounded raises rather than writing it: pass ``fast_dev_run``, or a ``limit_train_batches`` of your own, which is what ``trainer_args`` already forwards and what
:attr:`~pyhighlights.components.benchmarks.Benchmark.task_args` passes to a
whole grid at once.
A fraction of ``1.0`` is Lightning's own default and bounds nothing, so it is refused like an absent one.

:class:`~pyhighlights.components.tasks.GenSPPTask` is checked differently,
because its search runs outside Lightning and reads the whole split once per candidate: what bounds it is ``population_size`` and ``n_generations``, and a search of more than a handful of candidates is refused however the trainer was bounded.
A diagnosed search also marks every candidate and every generation, and scores one candidate at a time however many devices it was given: the stages report in the order they run, so two threads writing at once would record a model that never existed.

It is written through the standard library's ``logging`` under the ``pyhighlights.diagnostics`` logger, so a caller who wants the record on a console rather than in a file adds a handler to that logger and sets its level.
Nothing is formatted while nothing is listening.

API
---

.. automodule:: pyhighlights.utility.diagnostics
   :members:
