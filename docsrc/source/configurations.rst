Configurations
==============

``pyhighlights.configurations.keys`` holds every ``RegistrationKey`` the other
configuration modules register against, so model keys such as ``GRU_GENSPP``
and external trainer keys such as ``GRU_GENSPP_TRAINER`` are imported from one
place.

.. automodule:: pyhighlights.configurations.keys
   :members:

``pyhighlights.configurations.base`` holds the field sets model configurations
share. It registers nothing of its own, so a model inherits fields without
inheriting a registration.

.. automodule:: pyhighlights.configurations.base
   :members:

``pyhighlights.configurations.backbones``, ``losses`` and ``optimizers``
register the pieces the models are assembled from.

.. automodule:: pyhighlights.configurations.backbones
   :members:

.. automodule:: pyhighlights.configurations.losses
   :members:

.. automodule:: pyhighlights.configurations.optimizers
   :members:

One module per model registers its GRU and Transformer variants, along with
whatever pieces only that model uses -- ``genspp`` carries the frozen backbones
and the higher learning rate its genetic search needs.

.. automodule:: pyhighlights.configurations.fr
   :members:

.. automodule:: pyhighlights.configurations.genspp
   :members:

.. automodule:: pyhighlights.configurations.mgr
   :members:

.. automodule:: pyhighlights.configurations.mcd
   :members:

.. automodule:: pyhighlights.configurations.grat
   :members:

``pyhighlights.configurations.datasets`` registers the corpus loaders. ``R2A``
builds a Beer or Hotel loader; the aspect is a ``task`` variant.

.. automodule:: pyhighlights.configurations.datasets
   :members:
