Reproducing published results
=============================

pyhighlights ships tools; a paper is a set of values. Those values live in a
repository of their own, one per paper, so nothing here carries one study's
numbers and the library's releases are not tied to anyone's experiments.

A reproduction is a package of configurations and nothing else: every
component it names is the library's. It registers nothing on import, and a run
builds the registry over it with the library beside it.

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   import genspp2025
   from cinnamon.registry import Registry

   Registry.build(
       directory=Path(genspp2025.__file__).parent,
       external_directories=[Path(pyhighlights.__file__).parent],
   )

``directory`` is where registrations are executed and ``external_directories``
are indexed so their namespaces can be referenced -- the reproduction is what
runs, the library is what it points at. Building only the library leaves the
paper keys absent, which is the point.

The shape that has worked, and what the one below follows: a package per
corpus, one module per kind of thing registered, and its keys beside them in
``<corpus>/keys.py``. Neither corpus imports the other; a shared
``keys.py`` holds only what both need, the namespace and the seeds.

Each reproduction takes its own namespace, so two papers over one corpus can
prepare it differently without either becoming "the" version of it.

Written so far
--------------

**GenSPP, ACL 2025.** Ruggeri and Signorelli, *Interlocking-free Selective
Rationalization Through Genetic-based Learning* -- `paper
<https://aclanthology.org/2025.acl-long.59/>`_, `reference implementation
<https://github.com/nlp-unibo/gen-spp>`_. Two corpora against FR, MGR, MCD,
G-RAT and GenSPP, with the container and the Slurm jobs that run them:
`nlp-unibo/pyhighlights-genspp2025
<https://github.com/nlp-unibo/pyhighlights-genspp2025>`_. Its README carries
the settings it registers and, more usefully, every place it is *not* the
released implementation.

The corpus that reproduction reads is not a loader of its own. It is
:class:`~pyhighlights.components.loaders.ToyLoader` with a ``url``, and the
released pickle's older schema is handled by overriding ``parse`` in the
reproduction — fifteen lines, because a difference in serialisation is not a
difference in what the corpus is. See :doc:`datasets`.
