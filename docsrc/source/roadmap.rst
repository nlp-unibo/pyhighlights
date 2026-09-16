Roadmap
=======

What is being worked on, and what is known to be missing. Research directions
are discussed elsewhere until they are close enough to build.

In progress
-----------

**A study on legal text.** Terms of Service clauses, with no highlight
annotation at all: the question is whether a legal expert judges the predicted
highlights to be the right ones. It is the first use of the library from
outside it, so it is also what says whether the library is usable —
:class:`~pyhighlights.components.analyzers.PredictionAnalyzer`, the per-class
``weight`` on the classification criterion, ``encoder_lr``,
:class:`~pyhighlights.components.models.spp.implementations.StackedBackbone`,
``keep_checkpoints`` and the monitoring callbacks below all exist because that
study needed them.

Its first arm has run — four architectures over a held Legal-BERT, five
categories, five seeds — and reading it back is what found the two reporting
defects and the monitoring gap under *Done*. A study that only exercised the
library would not have found them; one that reported numbers did.

Done
----

**A reproduction is a repository.** ``pyhighlights_benchmarks`` is gone. A
paper's values live beside the container and the jobs that run them, in a
repository of their own, so the library ships tools and its releases are not
tied to anyone's experiments. The GenSPP 2025 reproduction is at
`nlp-unibo/pyhighlights-genspp2025
<https://github.com/nlp-unibo/pyhighlights-genspp2025>`_. What stays here is
the shape -- the two-directory registry build, a package per corpus, a
namespace per paper -- in :doc:`benchmarks`.

**One toy loader, which generates and reads.** A toy corpus is generated,
published, and read back by whoever reproduces what it produced; those were
two classes, ``ToyLoader`` and a ``GenSPPToyLoader`` that could not generate,
which is one object split by serialisation format rather than by what it is.
The released corpus was itself produced by this generator's ancestor, so the
split was never real.

:meth:`~pyhighlights.components.loaders.ToyLoader.save` writes what ``url``
reads, in this library's own columns, so **a published toy corpus is a URL and
a digest in a configuration** rather than a loader written per dataset. A saved
corpus carries its splits; a flat one is divided by ``train_ratio``,
``val_ratio`` and ``split_seed``, which is how the GenSPP baselines divide
theirs. :meth:`~pyhighlights.components.loaders.ToyLoader.parse` converts a
corpus older than these columns.

The rule the two classes were protecting is kept and tested: **a configured
source is never fallen back on.** A loader given a ``url`` it cannot read
raises, where generating instead would return a corpus of the right shape and
different content -- the one failure nothing downstream can see, because every
metric still computes.

``tools/build_datasets.py`` converts the release as it builds the artifact, so
the published record holds a corpus the loader reads directly. Checked against
the old loader over all ten thousand released rows: identical text, labels,
highlights and tokens across 6400 / 1600 / 2000.

**Precision, recall and how many spans.** The tp/fp/fn counters were already
kept and only their F1 was exposed. A sparsity target moves precision and
recall in opposite directions and their F1 hides it, so both are reported, and
they reach an empty denominator on *different* splits -- each says ``nan`` on
its own. :class:`~pyhighlights.utility.metrics.SelectionSpans` counts
contiguous runs, which contiguity being a penalty and never a reported number
had left unsaid: six words in one span and six scattered are the same
``selection_size`` and not the same highlight.

**A synthetic corpus is a control only while something checks that it is.**
:mod:`~pyhighlights.components.shortcuts` asks whether anything other than the
annotated evidence predicts the label. ``report()`` ranks every n-gram and
length threshold against a permuted control, since with thousands of features
the best of them beats the majority baseline by chance. ``check()`` is the
gate and runs on the **ablated** corpus: a scan of the corpus itself cannot
gate anything, because the annotated patterns are meant to predict and a
pattern shared by two of three classes still names the third by its absence.

Pointing it at this library's own toy corpus found three features that solved
the task at perfect accuracy without reading a character -- inserted patterns
made the document longer for the class with the longer pattern, unequal
pattern lengths made the *highlight* wider for it, and contamination drawn
only from other classes said which class a sample was not. All three are
fixed. It is not toy-specific: token n-grams over a corpus of words is the
question of whether punctuation predicts an unfair clause.

**A toy class is a conjunction, highlighted as spans.** A class's trigger is a
list of patterns, all of which must appear, and a pattern may belong to more
than one class -- so the gold highlight is disjoint spans rather than one run,
and no single pattern is sufficient. ``contaminations`` scatters proper chunks
of the patterns through the filler, without which a fragment classifies as
well as the pattern it came from.

**Knowledge grounding.** A corpus may explain its labels in free text rather
than in spans: ToS-100 records which legal rationales make each unfair clause
unfair, and never which words carry them. So *which* rationale applies is a
gold standard where the highlight is not.

:class:`~pyhighlights.components.models.spp.grounded.GroundedSPP` extracts a
highlight pair for every knowledge base entry — the words of the input matching
the entry, and the words of the entry matching the input — scores each pair
with an :class:`~pyhighlights.components.models.spp.grounded.SPPComparer`, and
names the subset the input instantiates. Two encoder passes per batch whatever
the base's size, because the base is a property of the corpus rather than of a
sample: it arrives once through
:meth:`~pyhighlights.components.models.base.Model.load_knowledge` and never
rides in a batch.

Almost none of it is a new contract. Conditioning is a concatenation, so an
ordinary :class:`~pyhighlights.components.models.spp.base.SPPSelector` at twice
the width is a conditioned one; the comparer emits two logits, so the gate is
the same activation a token selection uses and the link loss is an existing
criterion under a new binding; the sparsity penalty binds to the knowledge axis
unchanged.

Scored by per-link F1 micro **and** macro, exact-set match and empty-set
accuracy, and by the two rationale-level faithfulness terms — ablate the
entries the model named and see whether the prediction moves. Nobody has
reported the last two on this corpus.

**The highlight is the predictor's input, on every backbone.**
:class:`~pyhighlights.components.models.spp.implementations.StackedBackbone`
masked its transformer's attention and then handed the whole sequence to the
GRU above it. A transformer carries every position's own input forward through
the residual stream whether or not anything attended to it, so a dropped
subtoken still had a state, and a recurrent encoder carried it to every
position after it. Changing only the dropped words moved the predictor's pooled
state by 0.48; the GRU backbone, which zeroes its dropped embeddings, moved by
zero. The two implementations of one contract disagreed, which is what made it
a bug rather than a choice.

Every run over a stacked backbone before 0.8.0 is affected, including the legal
study's whole frozen arm.

**What a run is monitored by is a configuration.** A task used to take
``monitor`` and ``patience`` and build its own early stopping and checkpoint,
with ``mode`` fixed at ``min`` — so stopping on a *maximized* metric was not
expressible, and asking for it checkpointed the worst epoch. A task now takes
``callbacks``, a list of registration keys like its metrics and its losses, and
:mod:`pyhighlights.components.callbacks` registers what goes in it.

:class:`~pyhighlights.components.callbacks.GeneralizationLossScore` is there
because neither a loss nor a metric is the right thing to watch on its own.
Monitoring the validation loss stops a run with its rare-class F1 still
climbing; monitoring that F1 accepts a large loss regression, which on a
validation split of a few dozen positives is overfitting. The callback
combines them into one quantity,

.. math::

   \mathrm{score} = q - c \cdot \max\left(0, \frac{L}{L_{\mathrm{opt}}} - 1\right)

the penalty being Prechelt's generalization loss (1998, *Early Stopping — But
When?*) against the best validation loss so far. One quantity and not two
conditions, because early stopping decides when a run ends and the checkpoint
decides which epoch it is scored on: two quantities mean the reported model is
not the one the stopping rule chose, and ``results.json`` records scores rather
than the argument behind them. A task given callbacks that monitor different
quantities refuses to build.

:class:`~pyhighlights.components.callbacks.WarmupEarlyStopping` and
:class:`~pyhighlights.components.callbacks.WarmupModelCheckpoint` read
``warmup_epochs`` off the model and neither count nor checkpoint the epochs
before it. G-RAT pretrains a guider for ten epochs while the rationalizer sits
out, and a patience of five killed seeds before that pretraining finished —
which read as a diverging model rather than as a monitoring artifact. Two-phase
training, rather than a patience loose enough to cover both phases.

**Two metrics that were reported wrong.** ``selection_rate`` divided by the
padded batch width rather than by the document, so a corpus of short documents
in a widely padded batch reported roughly a third of what its selector kept.
The *training* objective was never affected — ``SparsityPenalty`` divides by
the real token count — so it was a reporting defect and nothing had to be
retrained. And a one-class F1 and a macro F1 were both registered under the
column name ``f1``, so only the manifest said which a table held; the two are
20 points apart on an imbalanced corpus, and a study read one against the
other's published numbers. A metric's column name is its own parameter, and
:class:`~pyhighlights.utility.metrics.ClassF1Score` says in its own docs which
class it scores.

**The library says highlight, in its arguments too.** Two public names
contradicted the convention the library states outright:
:class:`~pyhighlights.components.preprocessors.AnnotationAggregator` took
``rationale=`` and the constant behind it was ``RATIONALES``. They are
``highlights=`` and ``HIGHLIGHTS``. A third, ``rationale_losses``, is
``shared_losses`` — it holds the losses applied in *both* training phases
rather than the losses on the highlight, so the old name was wrong about more
than its vocabulary. **These are renames without shims**: a caller passing
``rationale=`` fails at the call.

What deliberately keeps the old word is everything naming something from
outside the library — the ``rationale`` column of the R2A files, HateXplain's
``rationales`` field, the *legal rationales* of the ToS knowledge base in
:mod:`~pyhighlights.components.models.spp.grounded`, and the literature's own
term where the docs quote it.

**The toy corpus was not a control, and now something checks that it is.** A
synthetic corpus earns its place by making the annotated evidence the only
thing that solves it. Nothing checked that, and three separate features solved
the toy task at perfect accuracy without reading a character: patterns were
*inserted*, so a class with a longer pattern produced a longer document; the
registered default's patterns had different lengths, so the highlight's own
width named the class; and contamination drawn only from other classes said
which class a sample was not.

:mod:`~pyhighlights.components.shortcuts` is the check. ``report()`` ranks
every n-gram and length threshold by how well it predicts, against a permuted
control — with thousands of features the best of them beats the majority
baseline by chance, so the threshold is what chance already offers. ``check()``
is the gate, and it runs on the **ablated** corpus: a scan of the corpus itself
cannot gate anything, because the annotated patterns are meant to predict and a
pattern shared by two of three classes still names the third by its absence.
With the evidence removed there is nothing left for any feature family to find,
which is why that claim is not bounded by one.

It is not toy-specific. Token n-grams over a corpus of words is the question of
whether punctuation predicts an unfair clause.

**A toy class is a conjunction, and its highlight is several spans.** A class's
trigger is a list of patterns, all of which must appear, and a pattern may
belong to more than one class — so the gold highlight is disjoint spans rather
than one run, which is the shape a real highlight has. ``triggers = ["aa",
"bc"]`` is the short spelling of a conjunction of one. ``contaminations``
scatters proper chunks of the patterns through the filler, without which a
fragment classifies as well as the pattern it came from: on 900 rows ``bc`` and
``abc`` both score 0.6667, and four contaminations leave the pattern there and
put the fragment at 0.5017.

Two changes to what the generator produces, so **a seed does not reproduce a
pre-0.9.0 corpus**: ``length`` counts the document rather than the filler, and
patterns overwrite filler instead of being spliced into it.

**Precision, recall and how many spans.** The tp/fp/fn counters were already
kept and only their F1 was exposed. A sparsity target moves precision and
recall in opposite directions and their F1 hides it, so both are reported;
they reach an empty denominator on *different* splits, and each says ``nan``
on its own.
:class:`~pyhighlights.utility.metrics.SelectionSpans` counts contiguous runs,
which contiguity being a penalty and never a reported number had left unsaid:
six words in one span and six scattered are the same ``selection_size`` and
not the same highlight.

**HateXplain reads its vocabulary from the vector file.** The released
baselines set ``use_pretrained_only=True``, under which their collator discards
the corpus and takes the whole of GloVe ``twitter.27B`` as its vocabulary, so
no evaluation token it covers is ever unknown. This reproduction fitted the
vocabulary on the training split, which embedded 5.35% of validation tokens
and 5.57% of test tokens as zero vectors. ``vocabulary_from`` names the choice.
**Every HateXplain number produced before this is superseded.** The registered
task also refuses to run without its vector file, rather than training on a
two-word vocabulary and reporting for it.

**One quantity under the name selection_rate.**
:class:`~pyhighlights.components.analyzers.HighlightPositionAnalyzer` pooled
its rate — every kept word over every word in the split — where
:class:`~pyhighlights.utility.metrics.SelectionRate` takes the mean of the
per-document rates. The two disagree whenever documents differ in length, and
one column name meant both.

**A candidate's fitness is a property of its chromosome.** The same chromosome
scored 2.8246, 2.4722 and 1.0000 within one GenSPP run: every evaluation built
a model and trained a predictor off the global torch random state, so each
shifted the next and the search ranked initialisations alongside genes.
``devices`` replaces ``device`` and scores candidates on several at once —
``["cpu"] * 8`` is the released implementation's thread pool, ``["cuda:0",
"cuda:1"]`` a node's cards.

**The toy corpus is made of characters, and its table is as wide as its
alphabet.** ``ToyLoader`` emitted word phrases in ``w17`` filler, which no
other part of this line of work reads; and a one-hot code over a
twenty-four-character alphabet needs twenty-four columns, where the release
uses 25 on its baselines and 26 on its genetic half, both leaving columns
nothing can set.

**One file per kind, one package per corpus.** ``toy.py`` and
``hatexplain.py`` held a loader, encoders, five architectures, a genetic
search, six tasks and a benchmark each. Each corpus is a package now, one
module per kind of thing it registers.

**Fewer knobs nobody turns, and names that say the number.**
``LeakageDetector.tolerance`` is gone — a shared row is leakage at any rate,
and ``report`` already says how much a corpus shares without refusing it. The
genetic search says how many couples it draws rather than hiding a selection
rate of 0.5 inside ``population_size // 2``, which read as though a generation
breeds half a population when it breeds a whole one.

**Compaction, and the channel it closes.** A select-then-predict model claims
the highlight *is* the predictor's input. The library enforced that a dropped
word cannot reach the predictor and not that the *shape* of the mask stays out,
which is a channel a selector can signal a label through: two clauses with the
same kept words in the same order but different gaps between them produced
different predictor inputs, by about a third of what changing a kept word does.

Two mechanisms, one per family. A
:class:`~pyhighlights.components.models.spp.implementations.GRUBackbone` steps
its recurrence over dropped positions, so their number changes the state; a
:class:`~pyhighlights.components.models.spp.implementations.TransformerBackbone`
re-indexes a kept word's position embedding when the gap before it changes.
Both are held by ``xfail(strict=True)`` tests.

``SPP(compact=True)`` gathers the kept positions into a shorter sequence and
closes both — measured on real Legal-BERT, 2.124514 in place against 0.000000
compacted. **Off by default and not a repair**: it changes what the predictor is
trained on rather than what it reads, and a compacted sequence can classify well
and mean nothing to a human reader. See *What the bottleneck does and does not
guarantee* under :doc:`models`.

**A tie is several labels sharing the top count.**
:class:`~pyhighlights.components.preprocessors.AnnotationAggregator` called a
tie when the top count was *one*, which coincides with a tie only at exactly
three annotators. HateXplain has three, so the defect was invisible everywhere
the library was exercised. Four annotators splitting 2-2 have a top count of
two, so no tie was detected and the label became whichever one ``Counter``
ordered first — silently, with ``ties="drop"`` not firing; a single-annotator
corpus has a top count of one on every row and emptied its splits entirely.
Nothing in the library's own corpora changes, which is the point: the component
stops being correct by coincidence.

**The ``cinnamon-core`` floor is 2.1.3**, and each step of it is a defect
rather than a preference. 2.1.1 stopped the registry scan walking a checkout's
own ``site-packages``. 2.1.2 validates a ``list`` or ``dict`` dependency's
members — below it ``validate_conditions`` recursed only when the whole field
was a ``Configuration``, so the registry dropped an invalid child and kept the
parent pointing at it, and every task here names its metrics, its callbacks and
its preprocessing steps as ``List[RegistrationKey[...]]``.

2.1.3 is the floor rather than 2.1.2 because 2.1.2 cannot run this library at
all. It forgot every module under a scanned root, and ``Registry.build`` is
given ``Path(pyhighlights.__file__).parent`` here — the package itself. Anything
imported before a build was dropped and re-imported by the registration
scripts, so the registry resolved a second copy of every class and a component
held across a reset failed ``issubclass`` against its own class, with a
``TypeError`` naming that class on both sides. 49 tests, one per test that
imports a component at module level. A build forgets only what it imported
itself now.

**A manifest cannot shadow its own key.** ``manifest.resolve`` wrote
``{"key": str(key), ...}`` over the resolved parameters, so a component with a
parameter *named* ``key`` — ``LeakageRemover`` has one — overwrote its
registration key with a column name. The field is ``@key``, which is not a
Python identifier and so cannot collide with any parameter; a manifest written
before the change is still read.

**Zenodo dataset artifacts.** Every loader downloaded a corpus as its authors
distributed it, so a reproduction depended on a URL somebody else controlled
and on splits nobody had fixed. The zero-leakage partitions are now published
as manifests — Beer at `10.5281/zenodo.22703544
<https://doi.org/10.5281/zenodo.22703544>`_, Hotel at `10.5281/zenodo.22711382
<https://doi.org/10.5281/zenodo.22711382>`_, ERASER ``movies`` at
`10.5281/zenodo.22711411 <https://doi.org/10.5281/zenodo.22711411>`_ — and the
GenSPP toy corpus at `10.5281/zenodo.22711449
<https://doi.org/10.5281/zenodo.22711449>`_, all CC-BY-4.0.

A toy corpus given a ``url`` fetches it instead of refusing, and
:class:`~pyhighlights.components.loaders.R2ALoader` and
:class:`~pyhighlights.components.loaders.ERASERLoader` pin the upstream
archives they were built against, so a changed upstream fails loudly rather
than being trained on. The manifests are a receipt rather than an input: a
pinned digest and a deterministic repair already give the same rows, and no
loader fetches a manifest to work.

Known gaps
----------

**A sparsity target is a limitation, not a missing parameter.** A single
``sparsity_threshold`` cannot serve a corpus whose train and test annotation
rates differ by a factor of three — ERASER ``movies`` annotates test at a 0.31
highlight rate against training's 0.09. Per-corpus targets are **not planned**:
only the training distribution is knowable at training time, and a target read
off test annotation is leakage. Gold length is not a rate either — on
``movies`` it scales as ``L**0.58``, between a fixed count and a fixed
share — so any threshold is one of two wrong rules, and which one it is
belongs in the write-up rather than in a knob.

**Supervised GenSPP.** ``GenSPPTask`` refuses ``highlight_supervision``, and
correctly: no gradient reaches the generator, so a supervision loss would
train nothing. Guiding a genetic search means conditioning the population it
draws from, which is a research question rather than a scheduled change.

**The rest of the metric suite.** Highlight F1, IoU, precision, recall,
selection rate, size and spans are registered, and faithfulness — sufficiency
and comprehensiveness — is measured over the test split. Four are still
missing, in the order they are worth having: **AOPC** sufficiency and
comprehensiveness, ERASER's bucketed form over k = 1, 5, 10, 20, 50%, which is
what would partly rescue a measure that a single threshold collapses;
**span-level IoU-F1**, ERASER's discrete match at IoU >= 0.5, which needs a span
representation and so is the same item as span selection rather than a second
one; **AUPRC** over soft token scores, which needs the pre-threshold scores
plumbed through; and ***text alone***, an independent bag-of-n-grams over the
highlighted words, which lives in a study rather than here.

**Layer-wise pruning has no contract.** PLMR and YOFO drop tokens *inside* a
pretrained language model's layers, which
:class:`~pyhighlights.components.models.spp.base.SPPBackbone`'s
``encode`` / ``pool`` / ``output_size`` cannot express — a backbone is asked
for representations, not for a schedule of what to discard at which depth.
Supporting either is a contract discussion before it is code.

Wanted, unscheduled
-------------------

**A second reproduction.** The shape is proven by ``genspp2025`` and the next
paper is mostly values. Read its released code rather than its PDF: the
learning-rate policy that matters is rarely the one in the text.

**More corpora.** Lei et al.'s beer with ``annotations.json`` — a separate
download from R2A's beer — and the query-based ERASER tasks, once there is
somewhere for a query to go.

Not planned
-----------

A combinatorial grid of every corpus against every model. A reproduction
states a published benchmark's configurations; the library stays a set of
tools, and a grid nobody published is a directory of numbers nobody asked
for.
