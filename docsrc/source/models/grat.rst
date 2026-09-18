G-RAT: guidance-based rationalization
=====================================

Hu and Yu, 2024, *Learning Robust Rationales for Model Explainability: A Guidance-Based Approach*, AAAI 2024, pages 18243-18251.
Reference implementation: https://github.com/shuaibo919/g-rat.

Problem
-------

The architectures before this one fight interlocking with the signal the pair already has, which is the label.
G-RAT observes that the label is a weak teacher for a selection: it says whether the words kept were sufficient, and never which words were worth keeping in the first place.
A model reading the whole document can answer the second question, and the paper's move is to build one and let it teach.

Method
------

A soft attention classifier over the full input is pretrained, then keeps training beside the rationalizer, and it guides the selection in two ways at once.
While its attention over the document supervises the selection directly, giving the selector a per-word target rather than a single label, its class distribution is matched against the rationalizer's so the two models agree about the document as well as about the words.
Since the guider reads everything, it is free of the bottleneck the rationalizer trains under, and its attention is a signal no highlight-only model can produce for itself.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"] --> GD["guider<br/>attention over every word"]
       GD -- "attention target" --> SEL["selector"]
       GD -- "class distribution" --> J{{"JSD term"}}
       X --> SEL --> H["highlight h"]
       H --> P["predictor"] --> Y["label"]
       P --> J

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef guide fill:#e8e4f3,stroke:#4a3b76,stroke-width:2px,color:#241a3d
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class SEL,P head
       class X,H,Y value
       class GD guide
       class J term

The two guidance terms are annealed against each other rather than both applied throughout.

.. math::

   \mathcal{L} = \mathcal{L}_{\text{cls}} + \lambda_s \Omega_s(h) + \lambda_c \Omega_c(h)
   + \alpha_t \, \mathcal{L}_{\text{guide}} + (1 - \alpha_t) \, \mathcal{L}_{\text{jsd}},
   \qquad
   \alpha_t = \max\!\big(1 - t \cdot \texttt{guide\_decay},\; 0\big)

Here :math:`t` counts the steps the rationalizer has taken.
Early in training the attention target carries the weight, since the selector has nothing of its own to go on; as :math:`\alpha_t` decays the distribution-matching term takes over, so the guider stops dictating the words and keeps agreeing about the label.

Training
--------

Two optimizers, and a rationalizer that sits out the first epochs.

1. The guider is stepped first, on its own classification loss over the full input.
2. The rationalizer selects and predicts, and the guider is read again in evaluation mode and under ``no_grad``, so the attention it supplies is a target rather than a path for gradient.
3. The guider's attention is folded onto the selection axis and normalised into a per-word target.
4. The five rationalizer terms are scored, with the guide and JSD terms scaled by the annealing factor.
5. The rationalizer's optimizer steps only once ``current_epoch >= pretrain_epochs``, so the first epochs train the guider alone.

That last point is the reason ``warmup_epochs`` exists on the model.
Until the rationalizer has taken a step the monitored quantities describe a model that has not moved, and monitoring from epoch zero stopped two of five seeds of one study inside that window.

Implementation
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The guider contract
     - :class:`~pyhighlights.components.models.spp.grat.GRATGuider`
   * - The attention classifier itself
     - :class:`~pyhighlights.components.models.spp.grat.AttentionGuider`
   * - The annealing factor
     - ``GRAT.guide_factor``
   * - Attention folded onto the selection axis
     - ``GRAT.to_selection_axis``, summed rather than averaged
   * - The per-word target
     - ``GRAT.guide_target``
   * - The staged loop
     - ``GRAT.training_step``
   * - Epochs the monitors should skip
     - ``GRAT.warmup_epochs``

The guider attends over subtokens, since that is what its encoder reads, while the selection it guides is over words, so each word takes the attention its subtokens hold between them.
The fold sums rather than averages, because attention is a distribution and averaging would report a long word as less attended than the short one beside it.
``AttentionGuider`` adds positive noise to its scores during training, which is the reference implementation's way of keeping the attention from collapsing onto a handful of words.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_GRAT``
     - :class:`~pyhighlights.configurations.grat.GRUGRATConfig`
   * - ``TRANSFORMER_GRAT``
     - :class:`~pyhighlights.configurations.grat.TransformerGRATConfig`
   * - ``GRU_GUIDER``
     - :class:`~pyhighlights.configurations.grat.AttentionGuiderConfig`
   * - ``TRANSFORMER_GUIDER``
     - :class:`~pyhighlights.configurations.grat.TransformerAttentionGuiderConfig`

Five loss terms rather than three, and a second list for the guider.

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_GRAT, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_GRAT, save_path="results", seeds=[0, 1])

   Registry.from_key(GRU_GRAT, pretrain_epochs=20, guide_decay=1e-3)

``pretrain_epochs`` defaults to ``10`` and ``guide_decay`` to ``1e-4``, which is a slow anneal: the guide term still carries most of the weight after a thousand steps.
``guide_loss`` and ``jsd_loss`` name the two terms the annealing scales, so a study registering its own criteria under different names says which ones they are.

API
---

.. automodule:: pyhighlights.components.models.spp.grat
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.grat
   :members:
