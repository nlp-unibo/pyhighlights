GenSPP: interlocking-free rationalization through genetic search
================================================================

Ruggeri and Signorelli, 2025, *Interlocking-free Selective Rationalization Through Genetic-based Learning*, ACL 2025.
Paper: https://aclanthology.org/2025.acl-long.59/.
Reference implementation: https://github.com/nlp-unibo/gen-spp.

Problem
-------

Every architecture on the preceding pages mitigates interlocking without removing its cause.
The cause is that one loss trains both modules at once, so the selector's gradient is computed through a predictor that the selector's own past choices shaped, and no amount of freezing, staging or regularising changes that the two are optimised together.
Interlocking is therefore a property of joint gradient descent rather than of any particular objective.

Method
------

GenSPP stops training the generator by gradient descent.
A genetic search runs over generator parameters, and each candidate is scored by training a fresh predictor on the selection it produces, so a candidate is judged by how well a predictor that owes it nothing can classify from its highlight.
Since no predictor is carried between candidates, there is no cooperative equilibrium to fall into, and the interlocking loop is not mitigated but absent.

.. mermaid::

   %%{init: {"theme": "base", "themeVariables": {"fontSize": "17px", "fontFamily": "Lato, sans-serif", "lineColor": "#37474f", "primaryTextColor": "#102027", "edgeLabelBackground": "#ffffff"}, "flowchart": {"nodeSpacing": 55, "rankSpacing": 70, "padding": 14, "curve": "basis"}}%%
   flowchart TD
       POP["population of generators"] --> C["one candidate"]
       C --> H["its highlight of the corpus"]
       H --> FP["a fresh predictor,<br/>trained from scratch"]
       FP --> F["fitness:<br/>selection rate against task loss"]
       F --> S["survivors, crossover, mutation"]
       S --> POP

       classDef shared fill:#cfe3ff,stroke:#1a4f9c,stroke-width:2px,color:#0b2545
       classDef head fill:#ffffff,stroke:#37474f,stroke-width:2px,color:#102027
       classDef value fill:#d7f0dc,stroke:#1e6b34,stroke-width:2px,color:#0d3018
       classDef term fill:#ffe9c7,stroke:#a35c00,stroke-width:2px,color:#3d2100
       class C,FP head
       class POP,H,S value
       class F term

The fitness trades what a candidate selected against what its predictor achieved.

.. math::

   \text{fitness} =
   \begin{cases}
     1.0, & \text{if } \mathcal{L}_{\text{task}} > \texttt{task\_loss\_limit} \\[4pt]
     \dfrac{1}{1 - \sqrt{(1 - r)(1 - \mathcal{L}_{\text{task}})}}, & \text{otherwise}
   \end{cases}

Here :math:`r` is the selection rate of the candidate's highlight and :math:`\mathcal{L}_{\text{task}}` is the cross entropy its freshly trained predictor reached.
The limit is what stops the search from buying a sparse selection with a model that has stopped classifying: above it a candidate scores the floor whatever it selected.
Below it both terms have to be small for the fitness to be large, so a candidate cannot win on sparsity alone.
The paper sets the limit per corpus, ``0.1`` on the toy corpus, which is nearly solved, and ``0.6`` on HateXplain, which is not.

Training
--------

The search is a trainer rather than a training step, and it owns the loop.

1. A founding population of ``population_size`` generators is drawn at random, each carrying its parameters as a chromosome.
2. Each candidate is scored on one device: its generator is frozen, a fresh predictor is trained on its selection for ``predictor_epochs``, and the fitness above is computed.
3. Survivors are chosen by half elitism, with the better half kept outright and the rest drawn in proportion to fitness.
4. Couples are crossed one-point and the children mutated by Gaussian noise on a share of their genes.
5. The generation is repeated until ``n_generations`` or until the best objective stops improving by ``stop_threshold``.

Gradient descent still happens inside step 2, and it only ever reaches the predictor.
Training a ``GenSPP`` model on its own therefore fits a predictor to whatever selection its untrained generator makes, which is why the search is the thing a study runs.

Implementation
--------------

Two classes rather than one: the model, and the search that fits it.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - What
     - Where
   * - The model
     - :class:`~pyhighlights.components.models.spp.genspp.GenSPP`
   * - The search
     - :class:`~pyhighlights.components.models.spp.genspp.GenSPPTrainer`
   * - Selection without the empty repair
     - ``GenSPP.select``, which scores an empty selection rather than repairing it
   * - Only the predictor is optimised
     - ``GenSPP.configure_optimizers``
   * - The generator held still while a predictor is fitted
     - ``GenSPP.on_train_epoch_start``
   * - The fitness
     - ``GenSPPTrainer.compute_fitness``
   * - Survivors, crossover, mutation
     - ``GenSPPTrainer._select_survivors``, ``_crossover``, ``_mutate``

``GenSPP`` refuses a configuration whose generator and predictor share parameters, since a search over generator parameters that also moved the predictor would be neither a search nor a training run.
The generator's modules are put in evaluation mode for the whole of a candidate's predictor fitting, because dropout inside a frozen generator would score the same candidate differently from one epoch to the next.
Candidates are evaluated one per device, and ``devices`` is the same knob for a pool of CPU workers and for a node's cards.

CPU workers are **processes**, CUDA workers are threads. A candidate is a small model, so its cost is the training loop stepping from Python rather than the arithmetic inside torch, and that loop holds the GIL: eight threads on eight cores were measured at 240% of a possible 800%. Processes lift that -- 1442 ms a candidate sequentially, 832 ms on eight threads, 293 ms on eight processes. CUDA is the other way round, since its kernels do release the GIL and a process per device would pay for a context each. A search falls back to threads where fork is unavailable, or where autograd has already run in the calling process, which torch refuses to combine with fork.

Why the pool is forked once, before the first candidate, rather than per generation::

   RuntimeError: Unable to handle autograd's threading in combination with
   fork-based multiprocessing.

Torch raises that once autograd has run threads in the parent, and it raises it in the child when the pass is attempted rather than at the fork.
So a pool is asked to train something trivial before it is trusted with a candidate, and one that cannot, or that does not answer within the probe timeout, is closed for threads.
``forkserver`` would be the start method that avoids forking a threaded process, and it cannot be used here: the registry is process-global state built once by the caller, and a worker that did not inherit it cannot build the model a chromosome is for::

   NotExpandedException: The registration graph has yet to be expanded!
   Configuration retrieval is not allowed.

Rebuilding it per worker would cost seconds each and register a second copy of every configuration.
Inheriting memory is what makes these workers correct, not merely cheap -- so the fork risk is bounded rather than removed.
A search opens its pool with one Python thread running, torch's being native, which is why CPython's own warning about forking a multi-threaded process does not fire outside a test runner that adds threads of its own.

One shared initial state is given to every candidate, so that a fitness is a property of a chromosome rather than of the predictor initialisation drawn alongside it.
The released implementation reaches the same place from the other side: it keeps a pool of models and resets each reused one to *that slot's* initial weights, which makes a candidate's predictor depend on the slot it was given.

Configuration
-------------

Two keys per backbone here, since the model and its search are registered separately.

.. list-table::
   :header-rows: 1
   :widths: 34 66

   * - Key
     - Configuration
   * - ``GRU_GENSPP``
     - :class:`~pyhighlights.configurations.genspp.GRUGenSPPConfig`
   * - ``GRU_GENSPP_TRAINER``
     - :class:`~pyhighlights.configurations.genspp.GRUGenSPPTrainerConfig`
   * - ``TRANSFORMER_GENSPP``
     - :class:`~pyhighlights.configurations.genspp.TransformerGenSPPConfig`
   * - ``TRANSFORMER_GENSPP_TRAINER``
     - :class:`~pyhighlights.configurations.genspp.TransformerGenSPPTrainerConfig`

The defaults are the release's:
``100`` generations over a population of ``50``, ``3`` predictor epochs per candidate, a mutation standard deviation of ``0.05``, and one CPU worker.

One deviation is deliberate.
The release mutates the selector's output bias at ``0.10`` while every other gene takes ``0.05``, which its paper does not report: the paper gives a single ``N(0.0, 0.05)``.
The default here follows the paper, and ``threshold_mutation_std`` sets the threshold's own deviation for a reproduction that needs the release's behaviour.
It names the deviation of the threshold rather than of one gene, since a head emitting two logits decides on their difference and two genes perturbed at ``s`` give that difference ``s * sqrt(2)``.

Which genes those are is asked of the selector rather than read off the end of the chromosome.
:meth:`~pyhighlights.components.models.spp.base.SPPSelector.threshold_parameters` returns them, and a selector that declares none is searched with a single deviation throughout.
:class:`~pyhighlights.components.models.spp.implementations.MLPSelector` declares the bias of its output layer; only parameters that land at the end of the flattened chromosome are counted, because the deviation is given to a trailing slice of it.

.. code-block:: python

   from pyhighlights.configurations.keys import GRU_GENSPP_TRAINER

   search = Registry.from_key(
       GRU_GENSPP_TRAINER,
       n_generations=50,
       population_size=20,
       devices=["cuda:0", "cuda:1"],
   )

A search of that size is expensive, and ``ToyGenSPPTrainerConfig`` exists for the case where the wiring rather than the result is what is being checked: two candidates and one generation, which finishes in seconds and means nothing.
The model's losses list holds the classification term alone, since the sparsity of a candidate is part of its fitness rather than a penalty on a gradient, and its backbones freeze their embeddings by default.

Running a search
----------------

GenSPP's generator is not trained: it is searched.
A population of generators is evolved, and a candidate is scored by fitting a predictor on the selections it makes, with the generator frozen, so the predictor never teaches the selector what to select, which is the cooperative equilibrium the other models have to fight.

.. code-block:: python

   from pyhighlights.configurations.keys import TOY_GENSPP_TASK

   Registry.from_key(TOY_GENSPP_TASK, seeds=[42]).run()

:class:`~pyhighlights.components.tasks.GenSPPTask` is an ``SPPTask`` in every
other respect: same corpus, preprocessing, metrics, seeds and output files.
Two things differ:

* It names a **search**, not a model. The model key is the search's own; naming
  it twice is a way for the two to disagree about which model was evolved. The
  search builds its own candidates, so a vector file named through
  ``embeddings`` reaches them through the search rather than through
  ``build_model``, which a searched model never goes through.
* A validation split is required. Fitness is task loss traded against selection
  rate, and both are measured there.

A generation draws ``int(selection_rate * population_size)`` couples by roulette wheel and crosses each into two children, so the default 0.5 adds one child per member: 25 couples and 50 children against a population of 50.
The children compete with their parents rather than replacing them, and survival is half elitism, the best half kept outright and the other half drawn from what is left, fitness-proportional and without replacement.
The population that comes out of a generation is the size that went in.

Each candidate's predictor is fitted by a throwaway Lightning trainer, so the inner training is the same code path every other model trains through, logging, checkpointing and sanity checks off, since a hundred generations build one trainer per candidate.
Gradients reach the predictor only:
``GenSPP.configure_optimizers`` hands over the predictor's parameters, and the generator is put back in evaluation mode at the start of every epoch so its dropout cannot score the same candidate two different ways.

Every candidate trains on the same batches in the same order.
The training loader shuffles, so the search draws one permutation from the seed before it starts and hands every candidate that: re-iterating the loader instead would give each candidate its own order, and a chromosome would score differently depending on how many candidates preceded it, or, with several devices, on how the workers interleaved.
The whole training split is held in memory for the duration of the search as a result.

Alongside the usual per-seed files, a GenSPP run writes ``best.ckpt``, the weights the search settled on, and ``search.json``, one entry per generation under ``training_progress``.
The entry is the best **objective** the search reached in that generation, ``1 / fitness``, so it falls as the search improves and it is what ``stop_threshold`` is compared against.
A search that stopped improving in its tenth generation and one still descending when the budget ran out report the same metrics otherwise.

API
---

.. automodule:: pyhighlights.components.models.spp.genspp
   :members:
   :show-inheritance:

.. automodule:: pyhighlights.configurations.genspp
   :members:
