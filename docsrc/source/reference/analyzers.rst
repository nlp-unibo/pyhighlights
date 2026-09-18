Analyzers
=========

An analyzer reads a results directory back and answers one question about it, returning a :class:`pandas.DataFrame` rather than printing, so the same analyzer serves a notebook, a test and a LaTeX table.

:class:`~pyhighlights.components.analyzers.MetricsAnalyzer`
   One row per task, one column per metric, ``mean +/- std`` across seeds. It
   walks every ``results.json`` beneath the directory, so it reads one task or
   a whole benchmark without being told which. A metric a task never measured
   reads as ``-``: a grid rarely reports the same set everywhere, and an
   unannotated corpus has no highlight F1 to give. ``pairs=True`` keeps the
   ``(mean, std)`` tuples, which
   :func:`~pyhighlights.components.analyzers.latex_table` renders as
   ``$12.34_{\pm 0.56}$``, escaping the underscores every metric name
   carries. Since a task keeps every run it has ever done, ``latest`` decides
   which the table is about: the newest run of each task by default, every run
   when asked. Runs are grouped by the name a result reports rather than by its
   directory, so a task that was renamed or moved is still the task its results
   say it is.

:class:`~pyhighlights.components.analyzers.HighlightPositionAnalyzer`
   Where in the document the selector looked, binned as a share of the
   document so lengths are comparable, and how much it kept. A selector that
   has learned nothing still selects something; position is what tells the two
   apart, since a model keying on the opening tokens of every document scores
   like one that found the highlight. A model with several selectors stores one
   mask per head; the analysis reads the head its aggregator keeps, which is
   the one every reported metric scored.

   Positions are word positions on either selection axis: a selection made
   over subtokens is folded through ``word_ids`` first, exactly as
   ``PredictionAnalyzer`` folds it, and a word split into several pieces is
   one word however many of its pieces were selected.

   ``absolute`` asks the other question. A model keying on the first three
   words of every document does that regardless of how long the document is,
   and a share hides it, in the first bin of a short document, and in the
   first tenth of a long one. Columns then cover the first ``bins`` words, and
   a selection past them counts in the total without a column of its own, so
   the shares sum to less than one by however much the tail holds.

:class:`~pyhighlights.components.analyzers.PredictionAnalyzer`
   What the selector kept, in words, one row per sample: the gold label and
   the predicted one, the words the run selected, and the highlight they spell
   out. A stored prediction is token ids and masks, enough to score, and
   unreadable on its own, so the corpus is reloaded and joined back to it.
   The run's ``manifest.json`` names the loader and the preprocessor that
   produced it, and those are the keys the analyzer builds: a corpus loaded
   from anywhere else is a different corpus. The corpus is not stored beside
   the predictions because it would be stored once per run and per seed.

   Selections are folded from token positions back to words through the
   ``word_ids`` the batch carries, so a subword model reports words like every
   other, and a word counts as selected when any of its subtokens was. A
   sample the corpus no longer holds is skipped rather than failing the split:
   a corpus that changed under a run is worth reporting around.

   This one resolves keys, so the registry has to be built before it runs,
   inside a cinnamon script it already is.

:class:`~pyhighlights.components.analyzers.LabelStudioExporter`
   The same rows, written where a domain expert can read them: one Label
   Studio file per run, pre-annotated with the words the model selected, so
   judging a highlight is a review rather than a fresh annotation. A corpus
   with no highlight annotation is exactly the case this is for, there is no
   F1 to report, and what the highlights are worth is a question for somebody
   who knows the domain.

   ``only`` narrows the export to samples of one class, which a corpus
   annotated for a rare one needs: the negatives are most of it and the
   interesting highlights are all on the positives. ``column`` decides which
   class that is, and the two answers ask different questions, ``"label"``
   takes the samples that *carry* the class and asks whether the model found
   the right words in them; ``"predicted"`` takes the samples the model
   *called* that class and asks whether the words it kept justify the call.
   The second is the one available on a corpus with no annotation to select
   by, and the one that surfaces a confident mistake. ``labels`` names the span
   label the file carries, and
   :func:`~pyhighlights.components.analyzers.label_studio` does the conversion
   on any frame carrying the columns
   :class:`~pyhighlights.components.analyzers.PredictionAnalyzer` reports.

   One ``label-studio-seed=<seed>.json`` per seed, beside the predictions it
   came from. A run's seeds are separate predictions of the same samples, so
   one merged file would show each sentence once per seed with different words
   marked, which is not a thing to read. Nothing to export writes no file
   rather than an empty one.

The ``run`` column of both prediction analyzers is a path under the directory they were pointed at, ``fr/2026-09-09T16-13-00`` rather than ``2026-09-09T16-13-00``, because a benchmark writes ``<name>/<started>`` per task, and two tasks that started inside the same second would otherwise report themselves as the same run.

None of them is interactive and none plots.
An analyzer that asks which folder you meant cannot run unattended, and a figure is a presentation choice that belongs to whoever is writing the paper.

API
---

.. automodule:: pyhighlights.components.analyzers
   :members:
   :show-inheritance:
