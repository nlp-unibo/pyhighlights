FR: folded rationalization
==========================

Liu, Wang, Wang, Li, Yue and Zhang, 2022, *FR: Folded Rationalization with a Unified Encoder*, NeurIPS 2022.
Reference implementation: https://github.com/jugechengzi/FR.

Problem
-------

In the standard select-then-predict pair the selector and the predictor own one encoder each, and the two encoders never read the same text: the selector encodes whole documents, since it has to decide about every word of one, while the predictor encodes only what the selector kept, which is a corpus of the selector's own construction and is off the distribution the selector sees.
The two representations drift apart.
Since the predictor's loss reaches the selector through a module that has learned to read something else, the gradient arriving at the selector is informative about the predictor's private corpus rather than about the document, and the pair is free to settle into the interlocking equilibrium described in :doc:`../concepts/select-then-predict`.
The argument of the paper is that a predictor holding a representation of its own can accommodate whatever subset it is handed, so the drift is not a side effect of interlocking but a condition for it.

Method
------

FR folds the two encoders into one.
While the selector and the predictor remain separate heads, they read the states of a single shared encoder, so the predictor's gradient reaches the selector through weights the selector itself uses, and neither module can build a representation the other does not share.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x<br/>w1 w2 ... wT"]
       X --> E1["encoder<br/>reads every word"]
       E1 --> SEL["selector"]
       SEL --> H["highlight h<br/>one decision per word"]
       X --> E2["encoder<br/>reads the selection"]
       H -. "zeroes the dropped words" .-> E2
       E2 --> P["predictor"]
       P --> Y["label"]
       E1 <== "one set of weights" ==> E2

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       class E1,E2 shared
       class SEL,P head
       class X,H,Y value

The objective is the one the base architecture already defines, with one classification term and the two penalties on the mask, and FR adds nothing to it.

.. math::

   \begin{aligned}
   \min_{\theta,\,\phi} \quad
     & \underbrace{\mathcal{L}_{\text{cls}}\big(p_\phi(y \mid h \odot x),\, y\big)}_{\text{predict the label from the highlight}} \\[6pt]
     & + \underbrace{\lambda_s \left| \frac{\sum_{i,t} h^{(i)}_t}{\sum_{i,t} m^{(i)}_t} - s \right|}_{\text{sparsity, over the batch}}
       + \underbrace{\lambda_c \operatorname*{mean}_{(i,t)\,\in\,\mathcal{V}} \left| h^{(i)}_t - h^{(i)}_{t-1} \right|}_{\text{contiguity}} \\[6pt]
   \text{subject to} \quad & \theta_{\text{enc}} = \phi_{\text{enc}}
   \end{aligned}

Every symbol, in order of appearance.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Symbol
     - Meaning
   * - :math:`x`, :math:`y`
     - One document of the batch and its gold label.
   * - :math:`h \in \{0,1\}^T`
     - The highlight: one binary decision per word, emitted by the selector.
   * - :math:`m \in \{0,1\}^T`
     - Which positions hold a real word rather than padding.
   * - :math:`h \odot x`
     - The document with every dropped word zeroed, which is the predictor's whole input.
   * - :math:`\theta`, :math:`\phi`
     - The selector's parameters and the predictor's, each an encoder part and a head.
   * - :math:`\mathcal{L}_{\text{cls}}`
     - Cross entropy of the predicted label against the gold one.
   * - :math:`s`
     - The target fraction of words to keep, ``0.15`` by default.
   * - :math:`\lambda_s`, :math:`\lambda_c`
     - The weights of the two penalties, ``1.0`` and ``2.0`` by default.
   * - :math:`\mathcal{V}`
     - The neighbouring pairs where both positions are real words, which is where a transition can be counted.
   * - :math:`\theta_{\text{enc}} = \phi_{\text{enc}}`
     - The folding: selector and predictor do not hold two encoders whose weights happen to agree, they hold one encoder.

Two details of the penalties are easy to read past.
The selection rate is one number for the whole batch rather than a mean of per-document rates, so a long document and a short one pull on it in proportion to their length, and a batch can meet the target by selecting sparsely in the long documents while the short ones keep everything.
The contiguity term averages over neighbouring pairs rather than summing, so its scale does not follow the length of the documents in the batch.

The highlight annotation appears nowhere above, so FR trains unsupervised with respect to the highlight and the annotation is used to score it afterwards.
Indeed, the whole intervention sits in that last constraint, and it costs one encoder's worth of parameters rather than two.

.. note::

   Those three criteria are what every model in the library optimises by default, and FR does not fix them.
   The target rate :math:`s`, the two weights, and the criteria themselves are fields of a configuration, so a study changes them by registering its own configuration rather than by editing the model.
   See :doc:`../reference/configurations`.

Training
--------

One optimizer, one step per batch, and no phases.

1. The shared encoder encodes the full input.
2. The selector reads those states and emits two logits per word.
3. During training a hard Gumbel-softmax turns the logits into a binary highlight and evaluation takes the argmax instead, and under either rule a sample that selected nothing keeps its highest-scoring word.
4. The same encoder encodes the input again with the dropped words masked out, and pools what remains.
5. The predictor reads the pooled vector and emits class logits.
6. The classification, sparsity and contiguity terms are summed, and one backward pass updates the encoder, the selector and the predictor together.

Implementation
--------------

The ``FR`` class holds no layer, no loss and no training logic of its own, since everything the architecture asks for is already what the base class does when it is given one encoder instead of two.
What is left is a constructor with two checks, and it is the whole file.

.. literalinclude:: ../../../pyhighlights/components/models/spp/fr.py
   :language: python
   :pyobject: FR

Given that the base class already reads a missing ``predictor_backbone`` as an instruction to reuse the selector's, the subclass exists to make that sharing mandatory rather than accidental, and to refuse a configuration whose key says FR while its fields describe an ordinary two-encoder rationalizer.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - Shared encoder is resolved
     - ``SPP.__init__``, the ``predictor_backbone is None`` branch
   * - Selector pass
     - ``SPP.select``
   * - Predictor pass over the highlight
     - ``SPP.predict``
   * - Loss terms
     - ``SPPModelConfig.losses``, the library default
   * - Refusal of a second encoder
     - ``FR.__init__``

Two invariants a reader can check directly: first, ``self.predictor_backbone is self.selector_backbones[0]`` holds for every FR instance, which is the architecture's whole claim written in one line; second, exactly one selector is allowed, since folding is defined for a pair and says nothing about what several selectors sharing one encoder would mean.
``tests/test_fr.py`` asserts both.

Configuration
-------------

Two keys, one per backbone.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_FR``
     - :class:`~pyhighlights.configurations.fr.GRUFRConfig`
   * - ``TRANSFORMER_FR``
     - :class:`~pyhighlights.configurations.fr.TransformerFRConfig`

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   from cinnamon.registry import Registry
   from pyhighlights.configurations.keys import GRU_FR, TOY_TASK

   Registry.build(directory=Path(pyhighlights.__file__).parent)

   task = Registry.from_key(
       TOY_TASK,
       model=GRU_FR,
       save_path="results",
       seeds=[0, 1],
       trainer_args={"accelerator": "cpu", "max_epochs": 1},
   )
   results = task.run()

Passing ``predictor_backbone`` to either key raises ``ValueError``, and deliberately so, since an FR run with two encoders is a different architecture reported under FR's name.
Still, three fields are worth setting: the sparsity target of the sparsity loss, the Gumbel temperature, and ``encoder_lr`` when a pretrained transformer is fine-tuned, since one rate cannot serve both a pretrained encoder and a selector initialized from scratch.

API
---

.. automodule:: pyhighlights.components.models.spp.fr
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.fr
   :members:
