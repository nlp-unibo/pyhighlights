DR: decoupled rationalization
=============================

Liu, Wang, Wang, Li, Qiu, Zhang, Han and Zou, 2023, *Decoupled Rationalization with Asymmetric Learning Rates: A Flexible Lipschitz Restraint*, KDD 2023, pages 1535-1547.
Reference implementation: https://github.com/jugechengzi/Rationalization-DR.

Problem
-------

Early in training the selector has learned nothing, so the text it hands the predictor is close to arbitrary.
The predictor is a capable model and it fits that text, which is degeneration: the predictor memorises an uninformative selection, reports a low loss on it, and tells the selector that the selection was good.
What the paper adds is a quantity behind the failure, since a predictor that memorises a bad selection is a predictor with a large Lipschitz constant, and restraining the constant stops the memorisation.

Method
------

DR restrains it by decoupling the two learning rates, and the restraint is written from the selection itself.
The selector trains at the optimizer's own rate, while the predictor trains at that rate multiplied by the fraction of the input the selection kept, recomputed from the mask every step.
A selector keeping a tenth of the text therefore trains its predictor ten times slower, and as the selection settles and becomes informative the factor rises on its own, so the restraint relaxes without a schedule anybody had to write.

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

The floor is the paper's and it matters more than it looks.
A selection keeping almost nothing would otherwise stop the predictor entirely, and a predictor that never moves cannot tell the selector which words were worth keeping.

The objective is unchanged from the base architecture, with one classification term and the two penalties on the mask.
Indeed, DR is the one architecture here that changes no loss at all: the whole method is in the optimizer.

Training
--------

One optimizer with two parameter groups, and the predictor's rate rewritten before every step.

1. The selector emits the highlight and the predictor classifies from it.
2. The classification, sparsity and contiguity terms are summed and backpropagated, as in FR.
3. The batch's selection rate is recorded outside the graph, since it sets a rate and never reaches a loss.
4. Before the optimizer steps, the predictor's groups are rescaled to the mean of the rates recorded since the last step.
5. The step is taken, with the selector at the optimizer's own rate and the predictor at the scaled one.

The mean rather than the last batch's rate is what makes gradient accumulation behave: several batches are selected before one update, and the rate that update is taken at is the mean of what they kept.

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
   * - The batch's selection rate
     - ``DR.selection_rate``, under ``no_grad``
   * - The rescale
     - ``DR.on_before_optimizer_step``
   * - Where the rate is recorded
     - ``DR.record``, training split only

Every rescale is written from the remembered base rather than from the current value, since scaling the current value would compound the factor batch after batch until the predictor stopped.
The groups are remembered by index rather than by reference, because ``Optimizer.load_state_dict`` replaces ``param_groups`` with fresh dictionaries and a resumed run would otherwise rescale objects the optimizer no longer owns.

Two limits worth knowing before reporting numbers.
DR owns the predictor's rate, so a learning-rate scheduler over that group would be overwritten at the next step, and nothing in the library configures one.
Under data parallelism each process scales by the selection rate of its own batch, which is a question a single-GPU reference implementation cannot answer.

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

``scale_floor`` defaults to ``0.05``, the paper's value, and ``predictor_backbone`` is required since a shared encoder would take both rates at once.
The reference implementation shares one embedding table between the two encoders and separates everything above it, whereas here a backbone owns its own table and the pair is separate throughout.

API
---

.. automodule:: pyhighlights.components.models.spp.dr
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.dr
   :members:
