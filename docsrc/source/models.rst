Models
======

Algorithms depend on ``SPPBackbone`` rather than GRU or Transformer details.
Output tensors retain a stable head dimension.

Base contracts
--------------

.. automodule:: pyhighlights.components.models.spp.base
   :members:
   :show-inheritance:

Algorithms
----------

.. automodule:: pyhighlights.components.models.spp.fr
   :members:

.. automodule:: pyhighlights.components.models.spp.genspp
   :members:

.. automodule:: pyhighlights.components.models.spp.mgr
   :members:

.. automodule:: pyhighlights.components.models.spp.mcd
   :members:

.. automodule:: pyhighlights.components.models.spp.grat
   :members:

.. automodule:: pyhighlights.components.models.spp.dr
   :members:

.. automodule:: pyhighlights.components.models.spp.mrd
   :members:

.. automodule:: pyhighlights.components.models.spp.dar
   :members:

Grounded in a knowledge base
----------------------------

A corpus may explain its labels in free text rather than in spans. Where it
does, a grounded model extracts a highlight pair for every knowledge base
entry -- the words of the input matching the entry, and the words of the entry
matching the input -- and names the subset the input instantiates.

.. automodule:: pyhighlights.components.models.spp.grounded
   :members:

Backends
--------

.. automodule:: pyhighlights.components.models.spp.implementations
   :members:
