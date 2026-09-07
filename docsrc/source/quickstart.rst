Quickstart
==========

Install core package:

.. code-block:: console

   pip install pyhighlights

Build registered GRU model from one Cinnamon key:

.. code-block:: python

   from pathlib import Path

   import pyhighlights
   from cinnamon.registry import Registry
   from pyhighlights.configurations.spp import GRU_FR

   Registry.build(directory=Path(pyhighlights.__file__).parent)
   model = Registry.from_key(GRU_FR)

Transformer backends require optional extra:

.. code-block:: console

   pip install "pyhighlights[transformers]"
