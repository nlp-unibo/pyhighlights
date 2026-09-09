Roadmap
=======

What is being worked on, and what is known to be missing. Research directions
are discussed elsewhere until they are close enough to build.

In progress
-----------

**Zenodo dataset artifacts.** Every loader downloads a corpus as its authors
distributed it, which means a reproduction depends on a URL somebody else
controls and on splits nobody fixed. Uploading verified artifacts turns
``GenSPPToyLoader``'s ``url=None`` into a real default and gives Beer, Hotel
and Movies their fixed zero-leakage splits. It is what stands between the
GenSPP reproduction and a run someone else can repeat.

**A study on legal text.** Terms of Service clauses, with no highlight
annotation at all: the question is whether a legal expert judges the predicted
highlights to be the right ones. It is the first use of the library from
outside it, so it is also what says whether the library is usable —
:class:`~pyhighlights.components.analyzers.PredictionAnalyzer` and the
per-class ``weight`` on the classification criterion both exist because that
study needed them.

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
