Datasets
========

A loader downloads a corpus once, hands back one :class:`pandas.DataFrame` per
split, and reports how much of one split another already contains.

.. code-block:: python

   from pyhighlights.components.datasets import R2ALoader

   loader = R2ALoader(task="hotel_Location")
   splits = loader.load()        # {"train", "val", "test"} -> DataFrame
   loader.check_leakage()        # raises if any split pair overlaps
   loader.datasets()             # -> HighlightDataset, ready for the collator

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

Leakage
-------

Published splits are not always disjoint, and a corpus that shares rows
between training and test reports highlight scores on examples the model was
trained on. Nothing downstream can detect that, so loaders address it directly:

``loader.leakage()``
   A frame with one row per ordered split pair: ``overlap`` rows of *right*
   found in *left*, and ``ratio``, that count over the size of *right*. Keys
   are whitespace- and case-normalised, since raw equality understates real
   overlap.

``loader.check_leakage(tolerance=0.0)``
   The same report, but raising when a pair exceeds the tolerance. Call it in a
   test or before a run.

``loader.duplicates()``
   Repeated rows inside each split.

``remove_leakage=True`` (the default)
   Splits are repaired at load time. Rows are walked in priority order —
   ``test``, then ``val``, then ``train`` — and each split keeps only rows no
   earlier split claimed and no earlier row of its own repeated. The annotated
   split comes first because it is the only one carrying highlights, so
   training and validation are what give rows up. :attr:`removed` records the
   count per split. Pass ``remove_leakage=False`` to reproduce a corpus exactly
   as distributed, leakage included.

Beer and Hotel (R2A)
--------------------

Multi-aspect BeerAdvocate reviews and TripAdvisor hotel reviews, from the R2A
release of Bao et al., 2018, *Deriving Machine Attention from Human
Rationales* — the archive the selective-rationalization literature (RNP, FR,
MGR, MCD, G-RAT) draws both corpora from.

:Download: ``https://people.csail.mit.edu/yujia/files/r2a/data.zip`` (162 MB)
:Tasks: ``beer0``, ``beer1``, ``beer2`` (appearance, aroma, palate);
        ``hotel_Location``, ``hotel_Service``, ``hotel_Cleanliness``
:Labels: binary
:Loader: :class:`pyhighlights.components.datasets.R2ALoader`
:Key: :data:`pyhighlights.configurations.datasets.R2A`, with the aspect as a
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

Measured with ``remove_leakage=False``. ``test ⊂ train`` is the share of the
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

With the default ``remove_leakage=True`` all of this is repaired: the annotated
split stays whole and the offending training and validation rows are dropped.
Validation is repaired before training, so it keeps its rows and training pays
for the overlap:

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
``check_leakage()`` passes for every task.
Numbers published on the distributed splits are reproducible with
``remove_leakage=False``, and are not comparable with numbers from the repaired
ones.

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
:Loader: :class:`pyhighlights.components.datasets.HateXplainLoader`
:Key: :data:`pyhighlights.configurations.datasets.HATEXPLAIN`

Both the label and the highlights are aggregated across annotators:

``label``
   Majority vote. 919 of the 20148 posts have all three annotators
   disagreeing, so no majority exists; ``ties="drop"`` removes them, as the
   paper does, and ``ties="keep"`` resolves them by annotator order.

``highlights``
   ``rationale="majority"`` (the default) keeps a token marked by more than
   half of the rationale vectors, ``"union"`` by any of them,
   ``"intersection"`` by all.

``normal`` posts carry no rationale by design, and 580 non-normal ones carry
none either. Both come back as all-zero highlights — "no token was marked",
not "not annotated" — so a highlight metric sees them as examples with no
positive tokens rather than skipping them.

The published splits share no post id, but they do share text: 6 test posts
and 3 validation posts also appear in training, and 28 training posts are
duplicates of each other. Small, but nonzero — and invisible to an id-based
check. The default repair removes 37 rows in total.

ERASER
------

Document classification with human evidence spans, from DeYoung et al., 2020,
*ERASER: A Benchmark to Evaluate Rationalized NLP Models*. A task ships a
``docs`` directory of whitespace-tokenized documents and one JSONL file per
split whose rows carry a ``classification`` and ``evidences`` — groups of
``[start_token, end_token)`` spans that become the highlights.

:Download: ``https://www.eraserbenchmark.com/zipped/<task>.tar.gz``
:Tasks: ``movies`` (1600 / 200 / 199 rows, 3.9 MB)
:Labels: binary (``NEG`` / ``POS``)
:Loader: :class:`pyhighlights.components.datasets.ERASERLoader`
:Key: :data:`pyhighlights.configurations.datasets.ERASER`

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
and that is all the default repair removes.

Toy
---

A synthetic corpus, generated in memory: each document is filler tokens with
one trigger phrase per class inserted at a random position, and the highlights
are exactly that trigger.

:Download: none
:Loader: :class:`pyhighlights.components.datasets.ToyLoader`
:Key: :data:`pyhighlights.configurations.datasets.TOY`

.. code-block:: python

   ToyLoader(sizes={"train": 64, "val": 16, "test": 16},
             triggers=("a great film", "a dull film"), seed=0)

Every split is annotated, a seed makes the corpus reproducible, and there is
nothing to fetch — which makes it the cheap way to exercise a model, a
configuration or a training loop before pointing it at a real corpus.

API
---

.. automodule:: pyhighlights.components.datasets
   :members:
   :show-inheritance:
