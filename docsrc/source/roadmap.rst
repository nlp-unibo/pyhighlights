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
:class:`~pyhighlights.components.analyzers.PredictionAnalyzer` and the
per-class ``weight`` on the classification criterion both exist because that
study needed them.

Done
----

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
