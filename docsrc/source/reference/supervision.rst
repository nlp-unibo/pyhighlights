Highlight supervision
=====================

Where a corpus annotates its **training** split, a task can train the selector against those annotations instead of leaving it to discover them:

.. code-block:: python

   Registry.from_key(TOY_TASK, highlight_supervision=True, highlight_coefficient=0.5)

The flag appends ``HIGHLIGHT_LOSS``, masked cross entropy over ``highlight_logits``, ``highlight_true`` and ``mask``, to the model the task names, weighted by ``highlight_coefficient``.
Nothing else changes: the same model key runs either way.

The two settings are different experiments, not two points on one scale.
The unsupervised one is the realistic problem; the supervised one is the ceiling it is measured against, and its ``val_loss`` carries a term the other has no equivalent for, so only the metrics compare.

**One annotation guides one head.** A model with several generators supervises the head its aggregator keeps, the one every reported metric scores, and leaves the rest to diverge, which is what those generators are there for.

**A corpus without training annotations is refused.** Unannotated positions are padded with ``-1`` and skipped by the criterion, so supervising a corpus annotated on test alone would train exactly as an unsupervised run does and report itself as that run's ceiling.
The task checks the training split and raises instead.

GenSPP refuses the flag outright: no gradient reaches its generator, so the loss would be built and train nothing.
Guiding a genetic search means conditioning the population it draws from, which is open work.
