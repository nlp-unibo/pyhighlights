Overview
========

What select-then-predict is
---------------------------

Most explainability methods answer the question *why did the model say that?*
after the fact: the model reads the whole input, produces an answer, and a
second procedure attributes that answer to parts of the input. The attribution
is a story about the model, and nothing forces the story to be true.

A select-then-predict model rearranges the pipeline so the question does not
arise. It has two parts:

- a **selector** — often called a generator — which picks a subset of the input
- a **predictor**, which sees *only* that subset and predicts from it

The selected subset is the **highlight**. Because the predictor never sees
anything else, the highlight is not an explanation of the prediction: it is the
input to it. A highlight that omits what mattered produces a worse prediction,
which is a property that can be measured rather than argued about.

The literature calls that subset a *rationale* and writes it ``r``; this
library says *highlight* and writes ``h`` everywhere — ``highlights`` in a
corpus frame, ``highlight_mask`` and ``highlight_logits`` in a batch,
``highlight_f1`` in a report. Same object, one word for it. Where a name comes
from outside — the ``rationale`` column of the R2A files, the per-annotator
vectors HateXplain ships — it keeps the name its source gave it.

The cost is that selection is a discrete choice inside a differentiable model,
which is what every architecture below is a different answer to.

The architectures
-----------------

All six are registered for both a GRU and a Transformer backbone. The
algorithms never mention either: a backbone is anything implementing
``encode`` / ``pool`` / ``output_size``, and swapping one for the other is a
key, not a code change.

**FR** — folded rationalization. Selector and predictor share one encoder, so
the predictor's gradient reaches the selector through shared weights and the
two cannot drift apart the way an independent pair does. Liu et al., NeurIPS
2022.

**MGR** — multiple generators, one shared predictor. Each generator proposes
its own highlight and the predictor sees all of them, which stops a single
degenerate generator from dictating the equilibrium. Inference reports one
head, since the generators converge. Liu et al., ACL 2023.

**MCD** — trained against both the selected input and the full input. A
predictor reading everything guides the generator: a highlight that
d-separates the label from the rest of the input makes the two predictions
agree. Liu et al., NeurIPS 2023.

**MRD** — maximizing the remaining discrepancy. Rather than asking the
highlight to predict the label, which a spurious feature answers just as well,
it asks what is left once the highlight is removed to stop looking like the
whole input: removing plain noise or a spurious feature leaves the remainder's
conditional distribution unchanged, and only the causal features move it. The
predictor is therefore trained on the complement and on the full input, never
on the highlight, and the generator maximizes the divergence between those two
predictions. Liu et al., NeurIPS 2024.

**G-RAT** — guider-regularized, staged. A soft attention classifier over the
full input is pretrained and keeps training alongside the rationalizer; its
attention supervises the selection and its predictions are matched in
distribution. Hu and Yu, AAAI 2024.

**GenSPP** — the generator is not trained by gradient descent at all. A genetic
search over generator parameters scores each candidate by training a fresh
predictor on it, which removes the cooperative equilibrium the others have to
fight. Ruggeri and Signorelli, ACL 2025.

The interlocking problem those six circle around is this: the selector and the
predictor are trained together, so a selector that has settled on the wrong
words trains a predictor to read the wrong words, which then reports that the
wrong words were the right ones. Every architecture above is a different way of
breaking that loop.

The corpora
-----------

Five loaders, each returning the corpus as its authors distributed it:

``beer``
   R2A beer reviews, three aspects: appearance, aroma, palate.

``hotel``
   R2A hotel reviews, three aspects: location, service, cleanliness.

``movies``
   The ERASER ``movies`` task — sentiment with evidence spans.

``hatexplain``
   Hate-speech posts with per-annotator token rationales.

``toy``
   Synthetic: one trigger phrase per class inside filler tokens.

A loader loads; it does not clean. What a study does to a corpus — resolving
HateXplain's annotator disagreement into a label, removing leakage, filtering
by length, remapping labels — is a :doc:`preprocessor <datasets>` it names, so
two studies over one corpus can prepare it differently without either becoming
"the" version of it.

Highlight annotation is optional and frequently absent. A corpus annotated on
test alone can be trained unsupervised and scored against its annotation; a
corpus with no annotation at all reports no highlight F1, and what its
highlights are worth is a question for somebody who knows the domain — which
is what :class:`~pyhighlights.components.analyzers.PredictionAnalyzer` is for.

What the library is, and is not
-------------------------------

pyhighlights ships **tools**: models, backbones, loaders, losses, metrics,
trainers, and the cinnamon registrations that let a project assemble them into
an experiment.

It does not ship an experiment. Which metrics to log, which aspect of Beer to
train on, which sparsity target to aim at, which embedding matrix to freeze —
all of that depends on a specific study, so all of it stays a configuration
the study writes. Published numbers live in ``pyhighlights_benchmarks``,
:doc:`beside the library rather than inside it <benchmarks>`, so nothing here
carries one paper's values.

Where to go next
----------------

- :doc:`quickstart` runs an experiment end to end and reads the results back.
- :doc:`tasks` is what a run is: the stages, what lands on disk, and the
  analyzers that read it.
- :doc:`models` and :doc:`data` are the contracts a new architecture or a new
  corpus is written against.
- :doc:`benchmarks` is how a paper's values are registered without putting
  them in the library.
- :doc:`roadmap` is what is being worked on.
