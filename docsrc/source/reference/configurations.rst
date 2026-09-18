Configurations
==============

``pyhighlights.configurations.keys`` holds every ``RegistrationKey`` the other configuration modules register against, so model keys such as ``GRU_GENSPP`` and external trainer keys such as ``GRU_GENSPP_TRAINER`` are imported from one place.

Every registration
------------------

The cards below are built from the registry itself, once per documentation build, so a key here is a key that exists and its defaults are the ones a run would get.
Each card carries the registration name, its tags as coloured chips, the component it builds, and every parameter with its type and default.
The search box filters by any of those, so ``transformer mcd`` narrows to the MCD registrations on a Transformer backbone and ``loader`` narrows to the corpora.

.. registry-cards::

Reading a card
--------------

A key is a name and a set of tags in a namespace, and the pair is what ``Registry.from_key`` resolves.
A card titled ``model`` with the tags ``fr`` and ``gru`` is the key ``pyhighlights.configurations.keys.GRU_FR``, and the constant is what a script should import rather than the string, since a typo in a constant is an ``ImportError`` and a typo in a string is a key that does not exist.

A parameter whose default reads as ``name[tag, tag]`` is itself a registration key, which is how a configuration names another one.
Writing registrations of your own, in a package beside the library rather than inside it, is :doc:`../tutorials/custom-model`.
A card marked **runnable** carries a ``run_method``, so ``cmn-run`` offers it from the command line.

API
---

.. automodule:: pyhighlights.configurations.keys
   :members:

.. automodule:: pyhighlights.configurations.base
   :members:

.. automodule:: pyhighlights.configurations.losses
   :members:

.. automodule:: pyhighlights.configurations.optimizers
   :members:

.. automodule:: pyhighlights.configurations.datasets
   :members:

.. automodule:: pyhighlights.configurations.preprocessors
   :members:
