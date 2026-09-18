Select-then-predict
===================

This page states the problem the library is about, the architecture that answers it, and the three difficulties every implemented model inherits from that architecture.
It assumes familiarity with supervised text classification and nothing else.
Read it before any model page: each one names a difficulty stated here and describes a different way of removing it.

The problem with post-hoc explanation
-------------------------------------

A classifier reads a document and returns a label.
Asked *why*, the usual answer comes from a second procedure that runs after the fact: attention weights, input gradients, or a surrogate fitted around the decision.
The attribution such a procedure returns is a claim about the model, and nothing in the pipeline forces that claim to be true, since the model computed its label from the whole input whatever the attribution says.
Two attribution methods disagreeing about one prediction is therefore an ordinary event rather than a bug, and the disagreement has no referee.

Select-then-predict removes the question instead of answering it better.

The architecture
----------------

A select-then-predict model factors the classifier into two modules trained together.

First, a **selector**, which the literature also calls a generator, reads the input and emits one binary decision per word.
Second, a **predictor** reads the words the selector kept, and nothing else, and returns the label.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027"}, "flowchart": {"nodeSpacing": 45, "rankSpacing": 55, "padding": 12}}%%
   flowchart TD
       X["input x<br/>w1 w2 w3 ... wT"]
       X --> SB["selector encoder"] --> SEL["selector<br/>two logits per word"]
       SEL --> H["highlight h<br/>one decision per word"]
       X --> PB["predictor encoder"] --> P["predictor"] --> Y["label y-hat"]
       H -. "masks the input of" .-> PB

       classDef enc fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       class SB,PB enc
       class SEL,P head
       class X,H,Y value

The subset the selector keeps is the **highlight**, written ``h``.
Because the predictor never sees the rest of the document, the highlight is not a story about the prediction: it is the input to it.
A highlight that omits what mattered produces a worse prediction, and that is a measurable property rather than an arguable one.
This is what the literature calls faithfulness by construction.

.. note::

   Most papers call this subset a *rationale* and write it ``r``.
   This library says *highlight* and writes ``h`` throughout, in the loaders,
   the models, the metrics and the batch fields, so a paper's ``r`` is this
   library's ``h``.
   Metric names stay as published, so a reported column still matches the
   paper it came from.

Formally, the selector defines a distribution over masks
:math:`p_\theta(h \mid x)` and the predictor a distribution over labels
:math:`p_\phi(y \mid h \odot x)`, and the pair is trained on the label alone:

.. math::

   \min_{\theta, \phi} \; \mathbb{E}_{h \sim p_\theta(h \mid x)}
   \big[ \mathcal{L}(p_\phi(y \mid h \odot x), y) \big]
   + \lambda_s \Omega_s(h) + \lambda_c \Omega_c(h)

The two penalty terms are explained below.
The highlight annotation, where a corpus ships one, appears nowhere in that objective: these models are trained unsupervised with respect to the highlight, and the annotation is used to score the highlight afterwards.

Difficulty 1: selection is discrete
-----------------------------------

The mask is binary, so the expectation above has no gradient to the selector by ordinary backpropagation.
Two answers are common in the literature, and this library takes the second.

The first answer is policy gradient, which treats the mask as an action and the predictor's loss as a reward.
It is unbiased and it has the variance such estimators are known for, which is why the original REINFORCE-based formulation is difficult to train.

The second answer is a continuous relaxation.
The library samples the mask with a hard Gumbel-softmax during training and takes the argmax during evaluation, so the forward pass is always a genuine binary mask while the backward pass runs through the relaxation.

.. code-block:: python

   # pyhighlights/components/models/spp/base.py
   def select_activation(self, highlight_logits):
       if self.training:
           return th.nn.functional.gumbel_softmax(
               highlight_logits, tau=self.temperature, hard=True, dim=-1
           )[..., 1]
       return highlight_logits.argmax(dim=-1).to(highlight_logits.dtype)

The ``temperature`` parameter is the :math:`\tau` of that relaxation, and it is a field of every model configuration.

Difficulty 2: an unconstrained highlight is the whole document
--------------------------------------------------------------

Nothing in the label loss prefers a short highlight, and keeping every word maximises the information the predictor receives.
The objective therefore carries two penalties on the mask itself.

**Sparsity** pushes the fraction of kept words towards a target rate, which is
a property of the study rather than of the architecture: the R2A corpora are conventionally run at ``0.15``, and a corpus whose evidence spans are longer needs a larger number.

**Contiguity** penalises transitions between neighbouring decisions, so a
highlight is a small number of spans rather than a scatter of isolated words.
It is what makes the selection readable.

.. code-block:: python

   # pyhighlights/utility/losses.py
   class SparsityPenalty(th.nn.Module):     # |rate(h) - threshold|
   class ContiguityPenalty(th.nn.Module):   # mean |h_t - h_{t-1}|

Both are registered losses and both sit in the default loss list of every model configuration, so a model that wants different weights changes two numbers rather than its code.

Difficulty 3: interlocking
--------------------------

This is the failure the implemented architectures exist to address, and it is worth stating carefully.

The selector and the predictor are trained together on one signal.
Early in training the selector has learned nothing, so it hands the predictor a nearly arbitrary subset, and the predictor learns to classify from that subset because it is the only input it has.
Once the predictor has learned to read the wrong words, the label loss reports that the wrong words are the right ones, so the selector is reinforced for producing them.
The pair converges to an equilibrium with high accuracy and a highlight that means nothing, and no term in the objective notices, because the objective only ever asked for the label.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "18px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027"}, "flowchart": {"nodeSpacing": 45, "rankSpacing": 55, "padding": 12}}%%
   flowchart LR
       S["the selector settles on<br/>uninformative words"] --> P["the predictor adapts<br/>to read them"]
       P --> L["the label loss is low,<br/>so nothing objects"]
       L --> S

       classDef step fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef bad fill:#fbd5d1,stroke:#a3231a,stroke-width:2px,color:#3d0d09
       class S,P step
       class L bad

Yu et al. named this *interlocking*.
Two related failures are usually discussed beside it.

**Degeneration** is the case where the predictor is strong enough to classify
from almost any subset, so the selector receives no pressure to improve.

**The private code** is the case where the selection stops carrying the
semantics of the input and starts carrying an encoding the predictor has learned to decode: accuracy is high, the highlight is unreadable, and the two facts are consistent.
The clearest form of it is signalling through the *shape* of the mask rather than the words in it, which :doc:`../reference/backbones` measures and discusses.

Every architecture in :doc:`../models/index` is a different way of breaking that loop.
They break it by sharing the encoder, by running several selectors, by adding a second predictor that never learned the code, by training the two halves at different rates, by predicting from the complement instead of the highlight, by supervising the selection with an external guider, or by abandoning gradient descent on the selector entirely.

How a highlight is scored
-------------------------

Three families of number are reported, and they answer different questions.

First, **task performance**: accuracy and F1 on the label, which say whether the bottleneck cost anything.

Second, **agreement with the annotation**, where a corpus ships one:
``highlight_f1``, ``highlight_iou``, ``highlight_precision`` and
``highlight_recall``, computed over words.
These require an annotated corpus and a study should say plainly that a corpus annotated on test alone was still trained unsupervised.

Third, **properties of the selection itself**:
``selection_rate``,
``selection_size`` and ``selection_spans``, which need no annotation and
report what the model actually did.
A selector that has learned nothing still selects something, so these numbers are how a degenerate run is recognised, together with
:class:`~pyhighlights.components.analyzers.HighlightPositionAnalyzer`, which
reports *where* in the document the selection fell.

Faithfulness diagnostics are computed by
:mod:`pyhighlights.components.faithfulness`: *sufficiency* compares the
prediction from the highlight against the prediction from the full input, and
*comprehensiveness* compares the full input against the input with the
highlight removed.

Where this lives in the code
----------------------------

One base class holds the architecture, and every model on the following pages is a subclass of it that changes one part.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Concept
     - Code
   * - The architecture
     - :class:`~pyhighlights.components.models.spp.base.SPP`
   * - Selector pass
     - ``SPP.select``
   * - Predictor pass over the highlight
     - ``SPP.predict``
   * - Predictor pass over the full input
     - ``SPP.predict_full``
   * - Predictor pass over the complement
     - ``SPP.predict_complement``
   * - Relaxed binary decision
     - ``SPP.select_activation``
   * - Encoder, backend-independent
     - :class:`~pyhighlights.components.models.spp.base.SPPBackbone`
   * - Per-word decision head
     - :class:`~pyhighlights.components.models.spp.base.SPPSelector`
   * - Label head
     - :class:`~pyhighlights.components.models.spp.base.SPPPredictor`
   * - What a forward pass returns
     - :class:`~pyhighlights.components.models.spp.data.SPPOutput`

A backbone is anything implementing ``encode``, ``pool`` and ``output_size``, and no algorithm mentions GRU or Transformer anywhere, so running an architecture on a different encoder is a registration key rather than a code change.

Where to go next
----------------

- :doc:`../models/index` compares the implemented architectures and links a page to each.
- :doc:`../tutorials/quickstart` runs one of them end to end on a synthetic corpus, and :doc:`../tutorials/custom-model` writes a method of your own.
- :doc:`../reference/backbones` is what the bottleneck does and does not guarantee, measured.
