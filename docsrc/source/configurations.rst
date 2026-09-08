Configurations
==============

``pyhighlights.configurations.spp`` registers GRU and Transformer variants for
FR, GenSPP, MGR, MCD, and G-RAT. Model keys such as ``GRU_GENSPP`` and external
trainer keys such as ``GRU_GENSPP_TRAINER`` are ``RegistrationKey`` entry
points.

.. automodule:: pyhighlights.configurations.spp
   :members:

``pyhighlights.configurations.datasets`` registers the corpus loaders. ``R2A``
builds a Beer or Hotel loader; the aspect is a ``task`` variant.

.. automodule:: pyhighlights.configurations.datasets
   :members:
