Writing your own method
=======================

:doc:`quickstart` runs one of the eight architectures the library ships.
This page is the next step: your own criterion, and your own architecture, registered beside the library rather than inside it and run through the same task.
Everything below is one worked example, in order, and it trains on the synthetic ``toy`` corpus in seconds.

The method is deliberately small, so that the mechanics stay visible.
It adds two things to an ordinary select-then-predict pair: a penalty on selections that fall in the opening words of a document, which is the degenerate behaviour :doc:`../reference/interlocking` warns about, and a consistency term asking the predictor to return the same label when part of the highlight is dropped.

Where the code goes
-------------------

A method is a package of its own, and the library is a dependency of it.
Nothing needs to be added to pyhighlights, and a study that forks the library to add a loss has made its results harder to reproduce rather than easier.

.. code-block:: text

   mymethod/
   ├── components/
   │   ├── criteria.py        # the penalty
   │   └── models.py          # the architecture
   └── configurations/
       ├── keys.py            # the keys, in a namespace of your own
       └── registrations.py   # what each key builds

One rule decides that layout: **the registry executes only the Python files under a directory named** ``configurations``.
A registration written anywhere else is never run, and the key it declares does not exist.
Components can live wherever you like, since a registration names them by import path.

A run builds your package with the library beside it.

.. code-block:: python

   from pathlib import Path

   import mymethod
   import pyhighlights
   from cinnamon.registry import Registry

   valid, invalid = Registry.build(
       directory=Path(mymethod.__file__).parent,
       external_directories=[Path(pyhighlights.__file__).parent],
   )

``directory`` is what runs and ``external_directories`` are indexed so their keys can be referenced, which is the same call :doc:`../reference/benchmarks` uses for a paper's reproduction.
Building the library alone leaves your keys absent.

A new criterion
---------------

A criterion is an ordinary ``th.nn.Module`` over tensors, and it knows nothing about batches, models or field names.

.. code-block:: python

   # mymethod/components/criteria.py
   import torch as th


   class LeadBiasPenalty(th.nn.Module):
       """How much of the selection falls in the opening of the document."""

       def __init__(self, head: float = 0.1):
           super().__init__()
           self.head = head

       def forward(self, selection: th.Tensor, mask: th.Tensor) -> th.Tensor:
           valid = mask.bool()
           width = selection.shape[-1]
           positions = th.arange(width, device=selection.device).expand_as(selection)
           lengths = valid.sum(dim=-1, keepdim=True).clamp_min(1)
           opening = (positions < (lengths * self.head).clamp_min(1)) & valid
           kept = (selection * opening).sum()
           return kept / selection.sum().clamp_min(1)

Two registrations turn that into something a model can be given, and the split between them is the point.
A ``criterion`` is the scoring object and a ``loss`` binds it to the fields it reads, so the same criterion scores different tensors by being registered twice with different ``inputs``.

.. code-block:: python

   # mymethod/configurations/registrations.py
   @register_class(
       name="criterion",
       tags={"lead_bias"},
       namespace=NAMESPACE,
       component="mymethod.components.criteria.LeadBiasPenalty",
   )
   class LeadBiasPenaltyConfig(Configuration):
       head: float = Param(0.1, gt=0.0, lt=1.0)


   @register_class(
       name="loss",
       tags={"lead_bias"},
       namespace=NAMESPACE,
       component="pyhighlights.utility.losses.Loss",
   )
   class LeadBiasLossConfig(Configuration):
       name: str = Param("lead_bias")
       loss: RegistrationKey[th.nn.Module] = Param(LEAD_BIAS_CRITERION)
       inputs: List[str] = Param(["highlight_mask", "mask"])
       coefficient: float = Param(1.0, ge=0.0)
       enabled: bool = Param(True)

``inputs`` names fields of the **namespace**, which is what the batch and the model's output look like to a loss: ``class_logits``, ``highlight_logits``, ``highlight_mask``, ``y_true``, ``mask``, ``highlight_true``, and whatever a model adds to it.
The names are positional, so ``["highlight_mask", "mask"]`` is what makes ``forward(selection, mask)`` receive the selection first.
``name`` is what the term is reported under, so this one appears as ``train_lead_bias`` and ``test_lead_bias`` in every results file.

A criterion registered this way needs no model of its own.
Adding ``LEAD_BIAS_LOSS`` to the ``losses`` list of any registered model is enough, and the rest of this page only writes an architecture because the second half of the method needs a second predictor pass.

A new architecture
------------------

Subclass :class:`~pyhighlights.components.models.spp.base.SPP` and override the one method the method actually changes.
Here that is ``compute_loss``, since the consistency term needs a prediction the base class does not make.

.. code-block:: python

   # mymethod/components/models.py
   class DropoutConsistencySPP(SPP):
       """A pair asked to predict the same label from a thinned highlight."""

       def __init__(self, drop_probability: float = 0.2, **kwargs):
           super().__init__(**kwargs)
           if not 0.0 < drop_probability < 1.0:
               raise ValueError("drop_probability must be in (0, 1)")
           if len(self.selectors) != 1:
               raise ValueError("DropoutConsistencySPP requires exactly one selector")
           self.drop_probability = drop_probability

       def thinned(self, highlight_mask: th.Tensor) -> th.Tensor:
           keep = th.rand_like(highlight_mask) >= self.drop_probability
           return highlight_mask * keep.to(highlight_mask.dtype)

       def compute_loss(self, input_data, output_data):
           head = self.aggregator(output_data)
           thinned_logits = self.predict(
               data=input_data, highlight_mask=self.thinned(head.highlight_mask)
           )
           values = self.head_namespace(
               input_data, output_data, thinned_class_logits=thinned_logits
           )
           return compute_losses(self.losses, values)

Four things in that class are worth naming, because every architecture in the library does the same four.

First, ``self.predict`` is how a model reads a selection, and passing it a different mask is how an extra pass is taken.
``predict_full`` and ``predict_complement`` are the two other passes the base class offers, and MCD and MRD are built out of exactly those.
Second, ``self.aggregator`` collapses the head axis, which is what makes a model with one selector and a model with several read the same way.
Third, ``head_namespace`` is what a loss binds to, and the keyword arguments handed to it are added to it, so ``thinned_class_logits`` becomes a field a registered loss can name.
Fourth, the constructor validates what the method assumes rather than trusting the configuration, which is why every library model refuses a shape it cannot mean.

The consistency term needs no new criterion, since a divergence between two class distributions is already registered:

.. code-block:: python

   @register_class(
       name="loss",
       tags={"consistency"},
       namespace=NAMESPACE,
       component="pyhighlights.utility.losses.Loss",
   )
   class ConsistencyLossConfig(LeadBiasLossConfig):
       name: str = Param("consistency")
       loss: RegistrationKey[th.nn.Module] = Param(
           RegistrationKey(name="criterion", tags={"js_div"}, namespace="pyhighlights")
       )
       inputs: List[str] = Param(["class_logits", "thinned_class_logits"])
       coefficient: float = Param(0.5, ge=0.0)

Then the model itself, which inherits every field an SPP model has from :class:`~pyhighlights.configurations.base.SPPModelConfig` and states only what differs:

.. code-block:: python

   @register_class(
       name="model",
       tags={"consistency", "gru"},
       namespace=NAMESPACE,
       component="mymethod.components.models.DropoutConsistencySPP",
   )
   class GRUConsistencyConfig(SPPModelConfig):
       name: str = Param("consistency")
       predictor_backbone: RegistrationKey = Param(GRU_BACKBONE)
       drop_probability: float = Param(0.2, gt=0.0, lt=1.0)
       losses: List[RegistrationKey[Loss]] = Param(
           [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS,
            CONSISTENCY_LOSS, LEAD_BIAS_LOSS]
       )

Running it
----------

The model is a key, and a task takes a model key, so nothing else has to be registered to run it.

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_TASK
   from mymethod.configurations.keys import GRU_CONSISTENCY

   task = Registry.from_key(
       TOY_TASK,
       model=GRU_CONSISTENCY,
       save_path="results",
       seeds=[0],
       trainer_args={"accelerator": "cpu", "max_epochs": 1},
   )
   results = task.run()

Both new terms are reported beside the library's own, since a loss reports under the ``name`` its registration gave it:

.. code-block:: text

   test_accuracy      0.8750      test_consistency   0.0224
   test_f1            0.8730      test_lead_bias     0.1250
   test_highlight_f1  0.4348      test_sparsity      0.1000

Registering a task of your own is what a study does next, so the corpus, the preprocessing and the seeds stop being keyword arguments and become part of the key.
:doc:`../reference/configurations` is the idiom, and :doc:`../reference/benchmarks` is a whole paper's worth of it.

A check that fails when the method breaks
-----------------------------------------

Two tests are enough here, and they are the two the method is: the thinning removes only selected words, and both new terms reach the loss.

.. code-block:: python

   @pytest.fixture(scope="module", autouse=True)
   def registry():
       Registry.build(
           directory=Path(mymethod.__file__).parent,
           external_directories=[Path(pyhighlights.__file__).parent],
       )


   def test_thinning_only_removes_selected_words():
       model = Registry.from_key(GRU_CONSISTENCY, drop_probability=0.5)

       highlight = th.tensor([[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 0.0]])
       thinned = model.thinned(highlight)

       assert th.all(thinned <= highlight)
       assert th.all(thinned[highlight == 0] == 0)


   def test_both_new_terms_are_scored():
       model = Registry.from_key(GRU_CONSISTENCY)
       batch = InputData(
           features=th.randint(1, 8, (4, 6)),
           mask=th.ones((4, 6)),
           sample_ids=th.arange(4),
           y_true=th.randint(0, 2, (4,)),
           highlight_true=th.full((4, 6), -1),
       )

       _, losses = model.compute_loss(batch, model(batch))

       assert "consistency" in losses
       assert "lead_bias" in losses

Build the registry once per module rather than once per test, as the library's own suite does.
A second ``Registry.build`` in one process rebuilds state the first one already holds, and the failure it produces names an unrelated component.

What to override for what
-------------------------

The base class is written so that a method changes one thing.
Where your method sits decides what you subclass and what you override.

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - The method changes
     - Override
   * - What the objective scores
     - ``compute_loss`` on an :class:`~pyhighlights.components.models.spp.base.SPP` subclass, as above, and register the terms it binds to. :doc:`../models/dar` is this with a frozen module added.
   * - How the two halves are optimised
     - ``configure_optimizers``, which :doc:`../models/mgr` uses for per-generator rates and :doc:`../models/dr` for a rate written per step.
   * - How the selection is made
     - ``select``, which :doc:`../models/genspp` overrides to score an empty selection rather than repairing it.
   * - The order the modules train in
     - Subclass :class:`~pyhighlights.components.models.spp.phased.PhasedSPP` and name the passes your criteria read, which is all :doc:`../models/mcd` and :doc:`../models/mrd` do.
   * - What reads the text
     - Implement :class:`~pyhighlights.components.models.spp.base.SPPBackbone`, which is ``encode``, ``pool`` and ``output_size``, and every architecture runs on it unchanged. See :doc:`../reference/backbones`.
   * - What a corpus is
     - Subclass :class:`~pyhighlights.components.loaders.HighlightLoader`, and keep what a study does to the corpus in a preprocessor instead. See :doc:`../reference/datasets`.

Two things not to override.
``forward`` is where the head axis is built, and a model that returns a different shape breaks every metric and analyzer downstream.
``namespace`` is what losses and metrics bind to, and the keyword arguments to ``head_namespace`` are the supported way to add a field to it.

A method written this way needs nothing from the library, which is the point of the layout.
Where a change to pyhighlights itself is what your method needs, a missing hook on the base class or a backbone the library should ship, :doc:`../project/contributing` is the environment, the checks and the branch flow.

