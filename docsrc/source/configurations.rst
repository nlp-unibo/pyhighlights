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

The classification criterion takes per-class ``weight`` values, which a
class-imbalanced corpus needs: 106 positives in 20,417 sentences is a corpus
answered correctly by a model that never predicts one. The weights are
declared rather than computed from the training split -- a fixed split has
fixed class frequencies, so the numbers are known before the run, and
declaring them puts them in the run's manifest where a weighting computed
inside the run would leave nothing.

.. code-block:: python

   class LTDCrossEntropyConfig(CrossEntropyConfig):
       weight: List[float] | None = Param([0.52, 12.4])

       @classmethod
       @register_method(
           name="criterion",
           tags={"cross_entropy", "ltd"},
           namespace="my-study",
           component="pyhighlights.utility.losses.CrossEntropy",
       )
       def default(cls):
           return super().default()

Weighted ``mean`` reduction divides by the sum of the weights in the batch
rather than by its size, so a weight sets how much a class counts relative to
the others and a batch of one class is unaffected by it.

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

``pyhighlights.configurations.datasets`` registers the corpus loaders, one per
corpus; the aspect of a Beer or Hotel loader is a ``task`` variant.

.. automodule:: pyhighlights.configurations.datasets
   :members:

``pyhighlights.configurations.preprocessors`` registers what runs after a
loader: the leakage detector, the preprocessing steps, and the pipelines that
compose them.

.. automodule:: pyhighlights.configurations.preprocessors
   :members:
