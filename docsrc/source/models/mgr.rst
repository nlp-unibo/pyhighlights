MGR: multi-generator rationalization
====================================

Liu, Wang, Wang, Li, Li, Zhang and Qiu, 2023, *MGR: Multi-Generator Based Rationalization*, ACL 2023, pages 12771-12787.
Reference implementation: https://github.com/jugechengzi/Rationalization-MGR.

Problem
-------

A single generator is a single point of failure.
Whatever subset it settles on early becomes the only text the predictor ever reads, so the predictor adapts to that subset and reports it as correct, which is the interlocking loop of :doc:`../concepts/select-then-predict` in its simplest form.
The paper puts a second failure beside it: a generator can collapse onto a subset the predictor happens to handle well, and once it has, nothing in the objective asks for anything else.

Method
------

MGR runs several generators against one shared predictor.
Each generator holds its own encoder and its own selection head, each proposes its own highlight of the same document, and the shared predictor is trained on all of them, so a degenerate generator is one voice among several rather than the only one.
Since the predictor has to classify from every proposal, it cannot specialise on the private subset of any single generator, and a generator proposing something uninformative is corrected by a predictor the others keep honest.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"]
       X --> G1["generator 1"] --> H1["highlight 1"]
       X --> G2["generator 2"] --> H2["highlight 2"]
       X --> G3["generator 3"] --> H3["highlight 3"]
       H1 --> P["shared predictor"]
       H2 --> P
       H3 --> P
       P --> Y["one label per highlight"]

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       class G1,G2,G3 shared
       class P head
       class X,H1,H2,H3,Y value

The learning rates are part of the method rather than a detail of it.
Generator :math:`i` trains at :math:`i \cdot \eta` and the predictor at :math:`\eta / n` for :math:`n` generators, so the generators move at rates that differ by design while the predictor moves slower than any of them, and a predictor that cannot chase a generator's proposal is a predictor that cannot interlock with it.

.. math::

   \mathcal{L} = \sum_{i=1}^{n} \Big[ \mathcal{L}_{\text{cls}}\big(p_\phi(y \mid h_i \odot x),\, y\big) + \lambda_s \Omega_s(h_i) + \lambda_c \Omega_c(h_i) \Big]

Here :math:`h_i` is the highlight of generator :math:`i`, and :math:`\Omega_s` and :math:`\Omega_c` are the sparsity and contiguity penalties written out on the :doc:`fr` page.
Every head is scored by the same three criteria, and what the sum over heads does is what ``loss_reduction`` controls: ``"sum"`` is the default and the reference implementation's, while ``"mean"`` divides by the number of heads so the total does not grow with it.

Evaluation reports one head.
The generators converge on the same selection, so reporting all of them would report one highlight several times, and ``inference_head`` names the generator that selects at validation and test time.

Training
--------

One optimizer holding one parameter group per module, and one step per batch.

1. Each generator encodes the document with its own encoder and emits its own highlight.
2. The shared predictor reads each highlight in turn and emits one set of class logits per head.
3. Every head is scored by the classification, sparsity and contiguity terms, and the totals are summed or averaged according to ``loss_reduction``.
4. One backward pass updates every generator and the predictor, each group at its own rate.
5. Validation and test call ``forward_one_head``, which selects and predicts with ``inference_head`` alone.

Implementation
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The per-generator rates
     - ``MGR.configure_optimizers``, the ``scales`` list
   * - One head at evaluation
     - ``MGR.forward_one_head``, ``validation_forward``, ``test_forward``
   * - Metrics scored on one head
     - ``MGR.update_metrics``
   * - Sum or mean over heads
     - ``MGR.compute_loss``
   * - Every head selected and predicted
     - ``SPP.forward``, which loops over the selectors

The rates are handed to ``build_optimizer`` as scales rather than written into the groups, so a model that also sets ``encoder_lr`` splits each group in two and keeps its own scale on both halves.
Three conditions are declared on the configuration rather than checked in the model: one backbone per generator, at least two generators, and an ``inference_head`` that exists.
Since the registry validates conditions while it expands keys, a grid over the generator count drops the impossible combinations before anything trains.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_MGR``
     - :class:`~pyhighlights.configurations.mgr.GRUMGRConfig`
   * - ``TRANSFORMER_MGR``
     - :class:`~pyhighlights.configurations.mgr.TransformerMGRConfig`

Three generators by default, each with an encoder of its own, and a fourth encoder for the shared predictor.

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_BACKBONE, GRU_MGR, MLP_SELECTOR, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_MGR, save_path="results", seeds=[0, 1])

   # five generators instead of three
   Registry.from_key(
       GRU_MGR,
       selector_backbones=[GRU_BACKBONE] * 5,
       selectors=[MLP_SELECTOR] * 5,
   )

``selector_backbones`` and ``selectors`` are lists here where every other architecture takes one of each, and the two lists have to stay the same length.
Unlike FR, ``predictor_backbone`` is required, since the generators share a predictor and that predictor encodes with an encoder of its own.

API
---

.. automodule:: pyhighlights.components.models.spp.mgr
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.mgr
   :members:
