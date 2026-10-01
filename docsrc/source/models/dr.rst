DR: decoupled rationalization
=============================

Liu, Wang, Wang, Li, Qiu, Zhang, Han and Zou, 2023, *Decoupled Rationalization with Asymmetric Learning Rates: A Flexible Lipschitz Restraint*, KDD 2023, pages 1535-1547.
Reference implementation: https://github.com/jugechengzi/Rationalization-DR.

Problem
-------

Early in training the selector has learned nothing, so the text it hands the predictor is close to arbitrary.
The predictor is a capable model and it fits that text, which is degeneration.
The predictor memorises an uninformative selection, reports a low loss on it, and tells the selector that the selection was good.
The paper relates the failure to a quantity: a predictor that memorises a bad selection has a large Lipschitz constant.
Restraining the constant stops the memorisation.

Method
------

DR restrains it by decoupling the two learning rates, and the restraint is written from the selection itself.
The selector trains at the optimizer's own rate.
The predictor trains at that rate multiplied by the fraction of the input the selection kept, recomputed from the mask every step.
A selector keeping a tenth of the text therefore trains its predictor ten times slower.
As the selection settles and becomes informative, the factor rises, so the restraint relaxes without a schedule.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart LR
       X["input x"] --> SEL["selector<br/>rate eta"] --> H["highlight h"]
       H --> R["selection rate<br/>kept / valid"]
       H --> P["predictor<br/>rate eta times the rate kept"]
       R -. "sets the rate of" .-> P
       P --> Y["label"]

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class SEL,P head
       class X,H,Y value
       class R term

.. math::

   \eta_{\text{selector}} = \eta,
   \qquad
   \eta_{\text{predictor}} = \eta \cdot \max\!\left( \frac{\sum_{i,t} h^{(i)}_t}{\sum_{i,t} m^{(i)}_t},\; \texttt{scale\_floor} \right)

The floor is the paper's.
A selection keeping almost nothing would otherwise stop the predictor entirely.
A predictor that never moves cannot tell the selector which words were worth keeping.

The objective is unchanged from the base architecture, with one classification term and the two penalties on the mask.
DR is the one architecture here that changes no loss, since the whole method is in the optimizer.

Training
--------

One optimizer with two parameter groups, and the predictor's rate rewritten before every step.

1. The selector emits the highlight and the predictor classifies from it.
2. The classification, sparsity and contiguity terms are summed and backpropagated, as in FR.
3. The batch's kept and valid token counts are recorded outside the graph, since they set a rate and never reach a loss.
4. Before the optimizer steps, the counts since the last step are summed over every process, and the predictor's groups are rescaled to their ratio.
5. The step is taken, with the selector at the optimizer's own rate and the predictor at the scaled one.

Counting every batch since the last step makes gradient accumulation behave: one update is taken at the rate of all the batches it accumulates.
Summing over processes makes data parallelism behave: every process applies the averaged gradient at the same rate.

Implementation
--------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The two groups, both built at the base rate
     - ``DR.configure_optimizers``
   * - The base rate remembered per group
     - ``DR.predictor_rates``, keyed by group index
   * - The batch's token counts
     - ``DR.selection_counts``, under ``no_grad``
   * - The rescale
     - ``DR.on_before_optimizer_step``
   * - Where the rate is recorded
     - ``DR.record``, training split only

Every rescale is written from the remembered base rather than from the current value.
Scaling the current value would compound the factor batch after batch until the predictor stopped.
The groups are remembered by index rather than by reference, because ``Optimizer.load_state_dict`` replaces ``param_groups`` with fresh dictionaries.
A resumed run would otherwise rescale objects the optimizer no longer owns.

DR owns the predictor's rate, so a learning-rate scheduler over that group would be overwritten at the next step.
Nothing in the library configures one.

Differences from the reference implementation
---------------------------------------------

The sparsity and contiguity penalties also differ from the reference implementation, as :ref:`penalty-differences` on the FR page describes.

Configuration
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Configuration
   * - ``GRU_DR``
     - :class:`~pyhighlights.configurations.dr.GRUDRConfig`
   * - ``TRANSFORMER_DR``
     - :class:`~pyhighlights.configurations.dr.TransformerDRConfig`

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_DR, TOY_TASK

   Registry.from_key(TOY_TASK, model=GRU_DR, save_path="results", seeds=[0, 1])

   Registry.from_key(GRU_DR, scale_floor=0.1)

``scale_floor`` defaults to ``0.05``, the paper's value.
``predictor_backbone`` is required, since a shared encoder would take both rates at once.
The reference implementation shares one embedding table between the two encoders and separates everything above it.
Here a backbone owns its own table, so the pair is separate throughout.

API
---

.. automodule:: pyhighlights.components.models.spp.dr
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.dr
   :members:
