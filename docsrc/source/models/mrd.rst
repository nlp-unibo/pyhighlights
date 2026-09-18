MRD: maximizing the remaining discrepancy
=========================================

Liu, Deng, Niu, Wang, Wang, Zhang and Li, 2024, *Is the MMI Criterion Necessary for Interpretability? Degenerating Non-causal Features to Plain Noise for Self-Rationalization*, NeurIPS 2024.
Reference implementation: https://github.com/jugechengzi/Rationalization-MRD.

Problem
-------

Maximum mutual information asks the highlight to predict the label, and a spurious feature correlated with the label answers that question just as well as a causal one.
Every architecture up to this point attacks the cooperation between the two modules, and none of them changes the question being asked.
MRD changes the question.

Method
------

Instead of asking what the highlight can say, MRD asks what is left once the highlight is removed.
Removing plain noise leaves the conditional distribution of the remainder unchanged, and removing a spurious feature leaves it unchanged as well, so only removing the causal features moves it, which makes a corpus full of spurious features behave like a clean one rather than needing a penalty for each pattern.
The generator therefore maximises the divergence between what the complement predicts and what the whole document predicts.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"] --> SEL["generator"] --> H["highlight h"]
       H -. "removed from x" .-> C["complement"]
       C --> PC["predictor<br/>reads the complement"]
       X --> PF["predictor<br/>reads every word"]
       PC --> D{{"discrepancy<br/>the generator maximises"}}
       PF --> D

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class SEL,PC,PF head
       class X,H,C value
       class D term

.. math::

   \begin{aligned}
     \text{predictor phase:} \quad & \min_{\phi} \;\; \mathcal{L}_{\text{cls}}\big(p_\phi(y \mid (1 - h) \odot x),\, y\big) + \mathcal{L}_{\text{cls}}\big(p_\phi(y \mid x),\, y\big) \\
     \text{generator phase:} \quad & \max_{\theta} \;\; \mathcal{D}\big(p_\phi(y \mid (1 - h) \odot x) \,\|\, p_\phi(y \mid x)\big) - \lambda_s \Omega_s(h) - \lambda_c \Omega_c(h)
   \end{aligned}

Two consequences make this model read oddly beside the others, and both are real rather than an artefact of the implementation.
The predictor never trains on the highlight at all: it is trained on the complement and on the full input, and the highlight pass exists only so the metrics have something to score.
The generator maximises a divergence rather than minimising one, which in the library is a loss registered with a negative coefficient.

Training
--------

The phases are MCD's, and only what each phase reads differs.

1. The predictor phase selects, detaches the selection, and trains the predictor on the complement and on the full input.
2. Both optimizers step, since the penalties on the mask are shared.
3. The generator phase selects again, freezes the predictor, and scores the remaining-discrepancy term.
4. The generator's optimizer steps alone.
5. The highlight pass runs under ``no_grad``, since nothing trains on it and it exists to be measured.

Implementation
--------------

Like MCD, MRD is a :class:`~pyhighlights.components.models.spp.phased.PhasedSPP` that names the passes its criteria bind to.

.. literalinclude:: ../../../pyhighlights/components/models/spp/mrd.py
   :language: python
   :pyobject: MRD

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The highlight pass, outside the graph
     - ``MRD.phase_class_logits``
   * - The complement and full-input passes
     - ``MRD.extra_logits``
   * - The complement itself
     - ``SPP.predict_complement``
   * - The alternation
     - ``PhasedSPP.training_step``

A highlight covering every valid word leaves the complement empty, which the backbones pool to zeros.
That is an honest reading of a model that kept everything rather than a case to repair, and it is why nothing guards it.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_MRD``
     - :class:`~pyhighlights.configurations.mrd.GRUMRDConfig`
   * - ``TRANSFORMER_MRD``
     - :class:`~pyhighlights.configurations.mrd.TransformerMRDConfig`

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Field
     - Default
   * - ``shared_losses``
     - sparsity and contiguity, scored in both phases
   * - ``predictor_losses``
     - classification over the complement, and classification over the full input
   * - ``generator_losses``
     - the remaining-discrepancy term

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_MRD, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_MRD, save_path="results", seeds=[0, 1])

API
---

.. automodule:: pyhighlights.components.models.spp.mrd
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.mrd
   :members:
