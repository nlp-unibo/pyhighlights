MCD: d-separation for causal self-explanation
=============================================

Liu, Wang, Wang, Li, Deng, Zhang and Qiu, 2023, *D-Separation for Causal Self-Explanation*, NeurIPS 2023.
Reference implementation: https://github.com/jugechengzi/Rationalization-MCD.

Problem
-------

Asking the highlight to predict the label rewards any subset that carries the label, and a spurious feature carries it as well as a causal one.
The usual answer is a penalty per known spurious pattern, which is a list somebody has to write and a corpus is free to fall outside of.
MCD asks for a property of the whole document instead: if the highlight really carries what determines the label, then knowing the rest of the document adds nothing once the highlight is known, which is conditional independence between the label and the remainder given the highlight.

Method
------

A second prediction is made from the full input, and the highlight is asked to make the two predictions agree.
Given a highlight that d-separates the label from the rest of the input, the full-input prediction and the highlight prediction are distributions over the same label with the same information behind them, so the divergence between them is the signal the generator is trained on and no list of spurious patterns is needed.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"] --> SEL["generator"] --> H["highlight h"]
       H --> PH["predictor<br/>reads the highlight"]
       X --> PF["predictor<br/>reads every word"]
       PH --> D{{"discrepancy<br/>the generator minimises"}}
       PF --> D

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class SEL,PH,PF head
       class X,H value
       class D term

Training alternates two phases, which is what makes the disagreement usable.
While the predictor phase trains the predictor on a selection it is handed and may not move, the generator phase trains the generator with the predictor frozen, so neither module can reduce the divergence by adapting to the other.

.. math::

   \begin{aligned}
     \text{predictor phase:} \quad & \min_{\phi} \;\; \mathcal{L}_{\text{cls}}\big(p_\phi(y \mid h \odot x),\, y\big) + \mathcal{L}_{\text{cls}}\big(p_\phi(y \mid x),\, y\big) \\
     \text{generator phase:} \quad & \min_{\theta} \;\; \mathcal{D}\big(p_\phi(y \mid h \odot x) \,\|\, p_\phi(y \mid x)\big) + \lambda_s \Omega_s(h) + \lambda_c \Omega_c(h)
   \end{aligned}

The predictor learns to classify from the highlight and from the whole document, the generator moves the highlight until those two answers agree, and the two penalties on the mask are scored in both phases.

Training
--------

Two optimizers and two forward passes per batch, driven by the model rather than by Lightning.

1. The predictor phase selects, detaches the selection, and scores the classification terms over the highlight and over the full input.
2. Both optimizers step, since the penalties on the mask are shared and reach the generator here as well.
3. The generator phase selects again, freezes the predictor's parameters, and scores the discrepancy term.
4. The generator's optimizer steps alone.

Two properties read oddly until they are checked against the reference implementation, and both are deliberate.
The shared criteria bind to the selection the generator produced rather than to the detached copy, so the generator takes two steps per batch on them and one on its own term, which is what ``train_util.train_decouple_causal2`` does upstream.
And each phase draws its own selection, since ``phase_forward`` selects once per phase, so the two phases of a batch optimise different masks of it.

Implementation
--------------

MCD is one of the two models built on :class:`~pyhighlights.components.models.spp.phased.PhasedSPP`, which owns the phases, the three loss lists and the manual training step.
What MCD adds is two methods naming the passes its criteria bind to.

.. literalinclude:: ../../../pyhighlights/components/models/spp/mcd.py
   :language: python
   :pyobject: MCD

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The alternation itself
     - ``PhasedSPP.training_step``
   * - Predictor frozen for the generator phase
     - ``PhasedSPP.generator_phase_loss``
   * - Detached selection for the predictor phase
     - ``PhasedSPP.phase_forward``
   * - The highlight pass
     - ``MCD.phase_class_logits``
   * - The full-input pass
     - ``MCD.extra_logits``, which exposes ``full_class_logits``

``predictor_phase`` is set to ``"classifier"``, which is the paper's name for that phase, so a training log reads as the published algorithm does.
Highlight supervision is refused rather than ignored: a supervision loss appended to the flat list would be dropped before the first batch, so it has to name the phase it belongs to and go in ``shared_losses``.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_MCD``
     - :class:`~pyhighlights.configurations.mcd.GRUMCDConfig`
   * - ``TRANSFORMER_MCD``
     - :class:`~pyhighlights.configurations.mcd.TransformerMCDConfig`

Three loss lists rather than one, and each names a phase.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Default
   * - ``shared_losses``
     - sparsity and contiguity, scored in both phases
   * - ``predictor_losses``
     - classification over the highlight, and classification over the full input
   * - ``generator_losses``
     - the discrepancy between those two predictions

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_MCD, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_MCD, save_path="results", seeds=[0, 1])

``predictor_backbone`` is required, since the highlight and the full input are read by an encoder of the predictor's own.

API
---

.. automodule:: pyhighlights.components.models.spp.mcd
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.components.models.spp.phased
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.mcd
   :members:
