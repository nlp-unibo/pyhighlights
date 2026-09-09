"""Reproductions of published results, kept out of the library proper.

pyhighlights ships tools; a paper is a set of values. Those values live here so
the library never carries one study's numbers, and this package is never
registered unless a caller asks for it:

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   import pyhighlights_benchmarks
   from cinnamon.registry import Registry

   Registry.build(
       directory=Path(pyhighlights.__file__).parent,
       external_directories=[Path(pyhighlights_benchmarks.__file__).parent],
   )

Each reproduction gets its own subpackage and its own namespace, so two papers
over the same corpus can disagree about how to prepare it.
"""

__all__: list[str] = []
