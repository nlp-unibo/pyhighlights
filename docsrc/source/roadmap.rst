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

``GenSPPToyLoader`` fetches its corpus instead of refusing, and
:class:`~pyhighlights.components.loaders.R2ALoader` and
:class:`~pyhighlights.components.loaders.ERASERLoader` pin the upstream
archives they were built against, so a changed upstream fails loudly rather
than being trained on. The manifests are a receipt rather than an input: a
pinned digest and a deterministic repair already give the same rows, and no
loader fetches a manifest to work.

Known gaps
----------

**Sparsity targets per corpus.** A single ``sparsity_threshold`` cannot serve
a corpus whose train and test annotation rates differ by a factor of three —
ERASER ``movies`` annotates test at a 0.31 highlight rate against training's
0.09. What the target should be a function of is an open question, not a
missing parameter.

**Supervised GenSPP.** ``GenSPPTask`` refuses ``highlight_supervision``, and
correctly: no gradient reaches the generator, so a supervision loss would
train nothing. Guiding a genetic search means conditioning the population it
draws from, which is a research question rather than a scheduled change.

**The full metric suite.** Faithfulness — sufficiency and comprehensiveness —
is measured over the test split. The wider set of highlight-based
explainability metrics is not yet registered.

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
