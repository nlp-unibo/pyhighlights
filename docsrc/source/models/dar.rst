DAR: discriminatively aligned rationalization
=============================================

Liu, Wang, Wang, Deng, Zhang, Wang and Li, 2024, *Enhancing the Rationale-Input Alignment for Self-explaining Rationalization*, ICDE 2024, pages 2218-2230.
Reference implementation: https://github.com/jugechengzi/dar.

Problem
-------

Selector and predictor are trained together on one signal, so nothing stops them from agreeing on a private code.
The selection drifts away from the semantics of the document, the predictor learns to read the drift, accuracy stays high, and the generator is rewarded for a highlight nobody else can interpret.
The paper calls that rationale shift, and it is the failure mode where every reported number looks correct and the highlight is unreadable.

Method
------

DAR adds a third module that never learned the code.
An aligner, which is a second predictor, is trained on the full input alone and then frozen, so its only notion of what a label looks like comes from ordinary text.
Since that module has never seen a highlight during its own training, it can only read one the way it reads text, and asking it to predict the label from the highlight therefore costs the generator anything it selected in a private code.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"] --> SEL["generator"] --> H["highlight h"]
       H --> P["predictor<br/>trained with the generator"] --> Y["label"]
       X -- "pretraining, then frozen" --> A["aligner<br/>reads full text only"]
       H --> A
       A --> AL{{"alignment term<br/>trains the generator"}}

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef frozen fill:#e8e4f3,stroke:#4a3b76,stroke-width:2px,color:#241a3d
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class SEL,P head
       class X,H,Y value
       class A frozen
       class AL term

.. math::

   \mathcal{L} = \underbrace{\mathcal{L}_{\text{cls}}\big(p_\phi(y \mid h \odot x),\, y\big)}_{\text{the pair, as usual}}
   + \lambda_s \Omega_s(h) + \lambda_c \Omega_c(h)
   + \underbrace{\mathcal{L}_{\text{cls}}\big(p_{\bar\psi}(y \mid h \odot x),\, y\big)}_{\text{the frozen aligner}}

The bar over :math:`\psi` is the point of the method: the aligner's parameters do not move while the rationalizer trains, so the fourth term trains the generator and nothing else.
An aligner that kept moving could co-adapt to the highlight, which is the one thing it exists not to do.

Training
--------

Pretraining first, then an ordinary single-optimizer loop.

1. Before the first rationalization epoch, the aligner is trained on the full input for ``pretrain_epochs`` epochs, in a loop the model drives itself.
2. Its parameters are frozen and a flag is written into a buffer, so a resumed run finds it trained rather than pretraining a second one.
3. Every batch afterwards selects, predicts from the highlight, and asks the frozen aligner to predict from the same highlight.
4. The four terms are summed and one backward pass updates the generator and the predictor, never the aligner.

Implementation
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The pretraining loop
     - ``DAR.pretrain_aligner``, called from ``on_train_start``
   * - The aligner reading a highlight
     - ``DAR.align``
   * - The aligner reading full text
     - ``DAR.align_full``
   * - The aligner left out of the optimizer
     - ``DAR.configure_optimizers``
   * - Pretraining survives a checkpoint
     - the ``aligner_ready`` buffer
   * - The alignment term
     - ``DAR.compute_loss``, which exposes ``aligner_class_logits``

The aligner is pretrained by the model rather than by the task, so it is part of the model, is checkpointed with it, and a resumed run finds it trained.
Two limits follow from that choice.
Under data parallelism each process pretrains its own aligner on its own shard with no gradient synchronisation, since the loop sits outside the strategy Lightning drives.
And an ``EarlyStopping`` callback counts epochs of the rationalizer rather than of the pretraining, which keeps a long pretraining off the patience counter.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_DAR``
     - :class:`~pyhighlights.configurations.dar.GRUDARConfig`
   * - ``TRANSFORMER_DAR``
     - :class:`~pyhighlights.configurations.dar.TransformerDARConfig`

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_DAR, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_DAR, save_path="results", seeds=[0, 1])

   Registry.from_key(GRU_DAR, pretrain_epochs=50)

``pretrain_epochs`` defaults to ``20``, where the reference implementation spends ``100`` and keeps the best of them against a validation split; this keeps the last, so the default is the smaller number a fixed budget can afford.
``aligner_backbone`` is the aligner's own encoder and its head is built from ``predictor``, since the two modules answer the same question about the same labels.
The alignment term is appended to ``losses`` by the model rather than declared in the list, so a registration cannot forget the term the method is.

API
---

.. automodule:: pyhighlights.components.models.spp.dar
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.dar
   :members:
