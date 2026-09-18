Contributing
============

This page is what a contributor needs in order to work on pyhighlights: the environment, the checks, the branch flow, and how a release is cut.

Setup
-----

.. code-block:: console

   git clone git@github.com:nlp-unibo/pyhighlights.git
   cd pyhighlights
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"

Transformer backbones are an extra, since a GRU run should not pull in ``transformers``:

.. code-block:: console

   pip install -e ".[dev,transformers,docs]"

Checks
------

``nox`` runs the same checks continuous integration runs, so a branch that passes locally passes there.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Command
     - What it checks
   * - ``nox``
     - Ruff and the test suite
   * - ``nox -s tests``
     - The tests, against a branch-coverage gate of 92%
   * - ``nox -s transformers``
     - The suite with the optional Transformers dependency installed
   * - ``nox -s docs``
     - The Sphinx build, with warnings treated as errors
   * - ``nox -s package``
     - The wheel and sdist build, and the metadata check

Run ``nox`` before opening a pull request, and the optional session that covers what the branch touched.

Branch and pull request
-----------------------

1. Branch from an updated ``main``, named ``feat/``, ``fix/``, ``docs/`` or ``chore/``.
2. Keep one logical change per branch, and add the tests and the public documentation in the same branch as the code.
3. Run ``nox``, plus the optional session the change calls for.
4. Open a pull request and complete the template.
5. Merge once ``All checks passed`` succeeds and the review is complete.
6. Prefer a squash or rebase merge, and delete the branch afterwards.

The ruleset ``main`` is configured with requires a pull request, one approval, resolved conversations, linear history and the ``All checks passed`` status check, and it blocks force pushes and branch deletion.

Documentation
-------------

The sources are under ``docsrc/source`` and the build is not committed.
``nox -s docs`` writes HTML to ``docsrc/build/html`` and treats every warning as an error, so a broken cross-reference fails the branch rather than the site.
Merges to ``main`` deploy that artifact through GitHub Pages.

Two conventions the pages hold to.
A model page carries the method beside the code that implements it, in the six sections :doc:`../models/fr` shows, and prose is written one sentence per source line so a diff names the sentence that changed.
Diagrams are written as ``mermaid`` directives rather than committed as images, which keeps a figure diffable and next to the prose it explains.

Release
-------

1. Update ``pyhighlights.__version__`` in a pull request and merge it.
2. Tag the matching commit: ``git tag -a vX.Y.Z -m "pyhighlights X.Y.Z"``.
3. Push the tag: ``git push origin vX.Y.Z``.
4. ``publish.yml`` verifies that the tag matches the version, builds the distributions, and publishes through PyPI Trusted Publishing.

Two repository settings stand behind that workflow, and they are the owner's to configure: GitHub Pages with **GitHub Actions** as its source, and PyPI Trusted Publishing for owner ``federicoruggeri``, repository ``pyhighlights``, workflow ``publish.yml``, environment ``pypi``.
