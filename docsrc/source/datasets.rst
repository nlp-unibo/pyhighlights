Datasets
========

A loader downloads a corpus once and hands back one :class:`pandas.DataFrame`
per split, **as distributed**. What is done to it next — repairing splits that
overlap, reducing several annotators to one judgement — is a preprocessing
step, because it is a decision about the study rather than about the corpus.

.. code-block:: python

   from pyhighlights.components.leakage import LeakageDetector
   from pyhighlights.components.loaders import HotelLoader
   from pyhighlights.components.preprocessors import LeakageRemover

   loader = HotelLoader(task="hotel_Location")
   splits = loader.load()                    # {"train", "val", "test"} -> DataFrame

   LeakageDetector().check(splits)           # raises: these splits overlap
   splits = LeakageRemover().process(splits)  # repaired, annotated split whole

Every loader produces the same five columns.

.. list-table::
   :header-rows: 1
   :widths: 15 15 70

   * - Column
     - Type
     - Meaning
   * - ``sample_id``
     - ``int``
     - Row index within its split, renumbered whenever rows are dropped
   * - ``text``
     - ``str``
     - The document as distributed
   * - ``tokens``
     - ``list[str]``
     - Whitespace split of ``text``; what ``highlights`` aligns to
   * - ``label``
     - ``int``
     - Class label
   * - ``highlights``
     - ``list[int]`` or ``None``
     - Per-token 0/1, ``None`` when the split carries no annotation

Preprocessing
-------------

:class:`~pyhighlights.components.preprocessors.Preprocessor` takes the splits a
loader produced and returns splits of the same shape, so any of them chains
with any other. Five ship with pyhighlights, and
:class:`~pyhighlights.components.preprocessors.Pipeline` runs a list of them in
order — its steps are registration keys, so a study states its pipeline in a
configuration instead of in code.

:class:`~pyhighlights.components.preprocessors.LeakageRemover`
   Walks the splits in priority order — ``test``, then ``val``, then ``train``
   — and keeps in each only rows no earlier split claimed and no earlier row
   of its own repeated. The annotated split comes first because it is the only
   one carrying highlights, so training and validation are what give rows up.
   ``priority`` is a parameter: reproducing a training set rather than an
   evaluation one is a different, equally legitimate choice. :attr:`removed`
   records the count per split.

:class:`~pyhighlights.components.preprocessors.AnnotationAggregator`
   Reduces per-annotator labels and rationales to one of each. See HateXplain
   below.

:class:`~pyhighlights.components.preprocessors.LengthFilter`
   Drops rows longer than ``max_length`` tokens, rather than truncating them:
   a truncated row keeps its label and loses the part of the annotation that
   fell off the end, which is then scored as if the model had missed it.
   :attr:`removed` records the count per split.

:class:`~pyhighlights.components.preprocessors.LabelMapper`
   Rewrites label values through a mapping — collapsing two classes into one,
   say. ``column`` may name the per-annotator judgements rather than a
   resolved label, and it usually should: a post the three annotators call
   ``hatespeech``, ``offensive`` and ``normal`` has no majority over three
   classes and a clear one over two, so folding the classes *before* the vote
   is counted and folding them after give different labels.

:class:`~pyhighlights.components.preprocessors.ClassWeights`
   Reads the class frequencies of one split -- ``train`` unless told otherwise
   -- and returns every row untouched. What it found is in :attr:`weights` and
   :attr:`counts`, and
   :class:`~pyhighlights.components.tasks.ClassWeightsTask` is what writes them
   somewhere they persist. Being a step rather than a calculation inside a task
   is what makes it run over the split the study trains on, after the filtering
   and the aggregation that change the frequencies.

:class:`~pyhighlights.components.preprocessors.LengthFilter` and
:class:`~pyhighlights.components.preprocessors.LabelMapper` ship without a
registration: ``max_length`` and a class mapping are study-specific numbers, and
inventing one in the library would make it look like a recommendation. The other
three are registered, since repairing leakage, reducing annotators and reading a
split's class frequencies are the same operation whoever asks for them.

Leakage
-------

Published splits are not always disjoint, and a corpus that shares rows
between training and test reports highlight scores on examples the model was
trained on. Nothing downstream can detect that, so
:class:`~pyhighlights.components.leakage.LeakageDetector` looks for it
explicitly. It holds no data: every method takes the splits to analyse, so the
same detector serves a loader's output and a preprocessed copy of it.

``detector.report(splits)``
   A frame with one row per ordered split pair: ``overlap`` rows of *right*
   found in *left*, and ``ratio``, that count over the size of *right*. Keys
   are whitespace- and case-normalised, since raw equality understates real
   overlap.

``detector.check(splits, tolerance=0.0)``
   The same report, but raising when a pair exceeds the tolerance. Call it in a
   test or before a run.

``detector.duplicates(splits)``
   Repeated rows inside each split.

Beer and Hotel (R2A)
--------------------

Multi-aspect BeerAdvocate reviews and TripAdvisor hotel reviews, from the R2A
release of Bao et al., 2018, *Deriving Machine Attention from Human
Rationales* — the archive the selective-rationalization literature (RNP, FR,
MGR, MCD, G-RAT) draws both corpora from.

:Download: ``https://people.csail.mit.edu/yujia/files/r2a/data.zip`` (162 MB),
           pinned by default at
           ``23fcb4cac883ec1de86d83a7747294d7fdae10061d3803fd4c34c930e66f25de``
           so a changed upstream fails loudly rather than being trained on.
           Pass ``sha256=None`` to skip the check
:Splits: the zero-leakage partition is published as manifests — Beer at
         `10.5281/zenodo.22703544 <https://doi.org/10.5281/zenodo.22703544>`_,
         Hotel at
         `10.5281/zenodo.22711382 <https://doi.org/10.5281/zenodo.22711382>`_.
         They are a receipt rather than an input: the digest above plus a
         deterministic repair already give the same rows
:Tasks: ``beer0``, ``beer1``, ``beer2`` (appearance, aroma, palate);
        ``hotel_Location``, ``hotel_Service``, ``hotel_Cleanliness``
:Labels: binary
:Loaders: :class:`pyhighlights.components.loaders.BeerLoader` and
          :class:`pyhighlights.components.loaders.HotelLoader`, both over the
          shared :class:`~pyhighlights.components.loaders.R2ALoader` parsing
:Keys: :data:`pyhighlights.configurations.keys.BEER` and
       :data:`pyhighlights.configurations.keys.HOTEL`, with the aspect as a
       ``task`` variant

**Annotation lives in a file named** ``train``. Inside the release,
``data/oracle/<task>.{train,dev}`` carry labels and text only, and the files
named ``.test`` carry **no rationale column at all**. The only per-token
annotation is the 200-row ``data/target/<task>.train``, which is what this line
of work reports highlight scores on. The default split map therefore reads:

.. code-block:: python

   {"train": "oracle/<task>.train",   # labels only, large
    "val":   "oracle/<task>.dev",     # labels only
    "test":  "target/<task>.train"}   # 200 rows, annotated

``splits`` is a parameter for anyone who reads the release differently.

Leakage in the distributed splits
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Measured on the splits as distributed. ``test ⊂ train`` is the share of the
annotated evaluation split found in training; ``val ⊂ train`` the share of
validation found in training.

.. list-table::
   :header-rows: 1

   * - Task
     - train
     - val
     - test
     - test ⊂ train
     - val ⊂ train
   * - ``hotel_Location``
     - 14472
     - 1812
     - 200
     - **1.000**
     - 0.184
   * - ``hotel_Service``
     - 101484
     - 12688
     - 200
     - **1.000**
     - 0.241
   * - ``hotel_Cleanliness``
     - 150098
     - 18764
     - 200
     - **1.000**
     - 0.241
   * - ``beer0``
     - 32276
     - 6392
     - 200
     - 0.445
     - 0.656
   * - ``beer1``
     - 28984
     - 5720
     - 200
     - 0.360
     - 0.659
   * - ``beer2``
     - 25748
     - 4994
     - 200
     - 0.415
     - 0.646

**Every Hotel aspect leaks its whole annotated evaluation split into
training.** All 200 annotated examples are also training examples, so any
Hotel highlight score published on these splits is measured on seen data. Beer
leaks less there but keeps roughly two thirds of its validation split inside
training. The splits also repeat rows internally — 1418 of the 14472
``hotel_Location`` training rows are duplicates.

A :class:`~pyhighlights.components.preprocessors.LeakageRemover` repairs all
of this: the annotated split stays whole and the offending training and
validation rows are dropped. Validation is repaired before training, so it
keeps its rows and training pays for the overlap:

.. list-table::
   :header-rows: 1

   * - Task
     - train
     - val
     - test
     - dropped from train
     - dropped from val
   * - ``hotel_Location``
     - 12546
     - 1767
     - 200
     - 1926
     - 45
   * - ``hotel_Service``
     - 84735
     - 12407
     - 200
     - 16749
     - 281
   * - ``hotel_Cleanliness``
     - 125868
     - 18390
     - 200
     - 24230
     - 374
   * - ``beer0``
     - 27973
     - 6388
     - 200
     - 4303
     - 4
   * - ``beer1``
     - 25139
     - 5720
     - 200
     - 3845
     - 0
   * - ``beer2``
     - 22436
     - 4994
     - 200
     - 3312
     - 0

Training loses 13-17% of its rows, the annotated split loses none, and
``LeakageDetector().check()`` passes for every task. Numbers published on the
distributed splits are reproducible by skipping the repair, and are not
comparable with numbers from the repaired ones.

One upstream artifact
^^^^^^^^^^^^^^^^^^^^^

Three rows carry one rationale flag more than their text has tokens —
``hotel_Location`` row 59, ``hotel_Cleanliness`` row 198, ``beer1`` row 119 —
and the surplus flag is always ``0``. An all-zero surplus is trimmed; any other
misalignment raises, since a real shift corrupts every label after it.

HateXplain
----------

Twitter and Gab posts labelled for hate speech, with token-level rationales
from three annotators. Mathew et al., 2021, *HateXplain: A Benchmark Dataset
for Explainable Hate Speech Detection*.

:Download: ``dataset.json`` (12 MB) and ``post_id_divisions.json`` from the
           ``hate-alert/HateXplain`` repository
:Rows: 20148 posts, 3 annotators each; splits come from the published
       ``post_id_divisions.json``
:Labels: ``hatespeech``, ``normal``, ``offensive``
:Loader: :class:`pyhighlights.components.loaders.HateXplainLoader`
:Key: :data:`pyhighlights.configurations.keys.HATEXPLAIN`

**The loader keeps every judgement.** ``label`` and ``highlights`` come back
unset and the raw material sits in ``annotator_labels`` and
``annotator_highlights``, two columns this corpus carries and the others do
not. Until an
:class:`~pyhighlights.components.preprocessors.AnnotationAggregator` has run,
the splits are not yet examples and ``datasets()`` says so rather than
guessing. There is no default aggregation, because reducing three annotators
to one judgement is exactly the choice two studies over this corpus make
differently:

``label``
   Majority vote. 919 of the 20148 posts have all three annotators
   disagreeing, so no majority exists; ``ties="drop"`` removes them, as the
   paper does, and ``ties="keep"`` resolves them by annotator order.

``highlights``
   ``rationale="majority"`` keeps a token marked by more than half of the
   rationale vectors, ``"union"`` by any of them, ``"intersection"`` by all.

:data:`~pyhighlights.configurations.keys.HATEXPLAIN_PIPELINE` chains the
aggregator with a
:class:`~pyhighlights.components.preprocessors.LeakageRemover`, in that order:
repairing leakage first would measure overlap over rows the tie handling then
removes.

``normal`` posts carry no rationale by design, and 580 non-normal ones carry
none either. Both come back as all-zero highlights — "no token was marked",
not "not annotated" — so a highlight metric sees them as examples with no
positive tokens rather than skipping them.

The published splits share no post id, but they do share text: 6 test posts
and 3 validation posts also appear in training, and 28 training posts are
duplicates of each other. Small, but nonzero — and invisible to an id-based
check. Repairing after aggregation removes 37 rows in total.

ERASER
------

Document classification with human evidence spans, from DeYoung et al., 2020,
*ERASER: A Benchmark to Evaluate Rationalized NLP Models*. A task ships a
``docs`` directory of whitespace-tokenized documents and one JSONL file per
split whose rows carry a ``classification`` and ``evidences`` — groups of
``[start_token, end_token)`` spans that become the highlights.

:Download: ``https://www.eraserbenchmark.com/zipped/<task>.tar.gz``, pinned by
           default at
           ``66e18d4e6c9df9e9f5544572b0bfe92a39673f74ecbfc3859b46cedb2f5b2dee``
           — the benchmark publishes no digest of its own. Pass ``sha256=None``
           to skip the check
:Splits: the zero-leakage partition is published as a manifest,
         `10.5281/zenodo.22711411 <https://doi.org/10.5281/zenodo.22711411>`_
:Tasks: ``movies`` (1600 / 200 / 199 rows, 3.9 MB)
:Labels: binary (``NEG`` / ``POS``)
:Loader: :class:`pyhighlights.components.loaders.MoviesLoader`, over the
         shared :class:`~pyhighlights.components.loaders.ERASERLoader` parsing
:Key: :data:`pyhighlights.configurations.keys.MOVIES`

**Only single-document tasks are supported.** A select-then-predict model
takes one token sequence and no query, so ``boolq``, ``esnli``,
``evidence_inference``, ``fever``, ``multirc`` and ``scifact`` are refused with
an explanation rather than silently folded into a document — pyhighlights has
nowhere to put a query yet. ``movies`` needs no such compromise.

Every split is annotated. Rows with an empty ``evidences`` list — one in
``movies`` — come back as all-zero highlights. The test split is annotated far
more densely than training (a 0.31 highlight rate against 0.09), since its
rationales aggregate several annotators; a sparsity target tuned on training
data is not tuned for it.

``movies`` has no cross-split leakage: one training row duplicates another,
and that is all a :class:`~pyhighlights.components.preprocessors.LeakageRemover`
removes.

Toy
---

A synthetic corpus, generated in memory: each document is filler tokens with
one trigger phrase per class inserted at a random position, and the highlights
are exactly that trigger.

:Download: none
:Loader: :class:`pyhighlights.components.loaders.ToyLoader`
:Key: :data:`pyhighlights.configurations.keys.TOY`

.. code-block:: python

   ToyLoader(sizes={"train": 64, "val": 16, "test": 16},
             triggers=("a great film", "a dull film"), seed=0)

Every split is annotated, a seed makes the corpus reproducible, and there is
nothing to fetch — which makes it the cheap way to exercise a model, a
configuration or a training loop before pointing it at a real corpus.

Corpus statistics
-----------------

:func:`pyhighlights.utility.statistics.describe` reports, per split, how long
the documents are and how much of them the annotation marks:

.. code-block:: python

   from pyhighlights.utility.statistics import describe

   describe(BeerLoader().load())

Document length sets a token budget — what
:class:`~pyhighlights.components.preprocessors.LengthFilter` drops rows over --
and ``highlight_rate`` is the ratio
:class:`~pyhighlights.utility.losses.SparsityPenalty` compares its
``threshold`` against. So a sparsity target is read off the **training**
split, and only off it. An evaluation split's rate is a fact to report once
the numbers are in, never a target: the model has not seen that split, and
tuning against it is tuning on the test set.

That leaves a real ceiling, and it belongs to the penalty rather than to any
corpus. One ``threshold`` names one corpus-level rate, so a corpus whose
splits are annotated at different densities — ERASER ``movies`` marks test at
0.31 against training's 0.09 — cannot be served by a single target, and no
choice of threshold fixes it. The mismatch is measured rather than hidden:
``selection_rate`` reports what the selector actually keeps at test, beside
the highlight scores — as a share of the document's own tokens, not of the
padded batch, so it is comparable between a corpus of short clauses and one of
long reviews.

API
---

.. automodule:: pyhighlights.components.loaders
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.leakage
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.utility.statistics
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.preprocessors
   :members:
   :show-inheritance:
