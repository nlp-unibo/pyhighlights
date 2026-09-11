Reproducing published results
=============================

pyhighlights ships tools; a paper is a set of values. Those values live in
``pyhighlights_benchmarks``, a package beside the library rather than inside
it, so nothing here carries one study's numbers and nothing registers unless
asked for.

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   import pyhighlights_benchmarks
   from cinnamon.registry import Registry

   Registry.build(
       directory=Path(pyhighlights_benchmarks.__file__).parent,
       external_directories=[Path(pyhighlights.__file__).parent],
   )

``directory`` is where registrations are executed and ``external_directories``
are indexed so their namespaces can be referenced — the reproduction is what
runs, the library is what it points at. Building only the library leaves the
paper keys absent, which is the point.

Each reproduction takes its own namespace, so two papers over one corpus can
prepare it differently without either becoming "the" version of it.

GenSPP, ACL 2025
----------------

Ruggeri and Signorelli, *Interlocking-free Selective Rationalization Through
Genetic-based Learning* — `paper <https://aclanthology.org/2025.acl-long.59/>`_,
`reference implementation <https://github.com/nlp-unibo/gen-spp>`_. Namespace
``genspp2025``. Two corpora against FR, MGR, MCD, G-RAT and GenSPP.

.. code-block:: python

   from pyhighlights_benchmarks.genspp2025.configurations.keys import TOY_BENCHMARK

   Registry.from_key(TOY_BENCHMARK, save_path="results").run()

What the released implementation sets, and what is therefore registered:

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * -
     - Toy
     - HateXplain
   * - Corpus
     - the released ``toy_dataset.pkl``
     - posts of 30 tokens or fewer
   * - Classes
     - three hidden patterns
     - two: ``offensive`` folded into ``hatespeech``
   * - Tokens
     - characters, 27-row table
     - words, GloVe ``twitter.27B`` 25d, frozen
   * - Backbone
     - GRU, hidden 8
     - GRU, hidden 16
   * - Sparsity
     - target 0.15, contiguity 2.0
     - target 0.22, no contiguity
   * - G-RAT
     - guide 1.0, JSD 1.0
     - guide 2.5, JSD 1.5
   * - GenSPP
     - expected cross entropy 0.1
     - expected cross entropy 0.6

Shared by both: Adam at 1e-3, batches of 64, up to 500 epochs, early stopping
on validation loss after 30 worse ones, and the seeds
``[2023, 15451, 1337, 2001, 2080]``. GenSPP searches 100 generations over a
population of 50, mutating every gene with Gaussian noise at ``0.05``, and
fits each candidate's predictor for 3 epochs at 1e-2.

Two things it needs from you
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**GloVe.** The vectors are a 1.4 GB download the paper expects you to fetch
yourself, so the task takes a path rather than a URL::

    Registry.from_key(HATEXPLAIN_FR_TASK, embeddings="glove.twitter.27B.25d.txt")

**The toy corpus.** ``GenSPPToyLoader`` fetches the published artifact,
`10.5281/zenodo.22711449 <https://doi.org/10.5281/zenodo.22711449>`_, released
under CC-BY-4.0 by both authors of the paper, and verifies its digest before
reading. The record holds the artifact rather than a loose pickle, since the
artifact is what carries the manifest, the licence and the citation alongside
the data; the loader unpacks it. Pass ``url=`` to read a local copy of the
archive or a local ``toy_dataset.pkl`` instead. With ``url=None`` it refuses
rather than synthesising a corpus of the same shape but different content,
which :class:`~pyhighlights.components.loaders.ToyLoader` would happily do.

Where this is not the paper
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **HateXplain is parsed from upstream**, from the released ``dataset.json``
  and the official ``post_id_divisions.json``, where the paper reads its own
  preprocessed pickles. The preparation is reproduced; the row counts may
  differ by whatever those pickles did that the JSON does not say.
* **One split scheme for all five models.** The released baselines and the
  released genetic code split the toy corpus differently. The baselines' —
  the first 80% train, a fifth of it held out for validation, the rest test —
  is used throughout, so the five numbers are comparable to each other.
* **Mutation is the paper's, not the release's.** Gaussian noise at ``0.05``
  for every gene; the released split mutator applies ``0.10`` to the final
  gene alone, which the paper does not describe.

API
---

.. automodule:: pyhighlights_benchmarks.genspp2025.corpora
   :members:
   :show-inheritance:

.. automodule:: pyhighlights_benchmarks.genspp2025.configurations.common
   :members:

.. automodule:: pyhighlights_benchmarks.genspp2025.configurations.toy
   :members:

.. automodule:: pyhighlights_benchmarks.genspp2025.configurations.hatexplain
   :members:
