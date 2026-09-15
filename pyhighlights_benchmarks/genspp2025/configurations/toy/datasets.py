"""The released ``toy_dataset.pkl``, as this reproduction reads it."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class

from pyhighlights.configurations.datasets import LoaderConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)


@register_class(
    name="dataset",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights_benchmarks.genspp2025.corpora.GenSPPToyLoader",
)
class ToyConfig(LoaderConfig):
    """The released ``toy_dataset.pkl``, not a regenerated corpus."""

    url: str | None = Param(None)
    sha256: str | None = Param(None)
    train_ratio: float = Param(0.8, gt=0.0, lt=1.0)
    val_ratio: float = Param(0.2, ge=0.0, lt=1.0)
    split_seed: int = Param(15000)
