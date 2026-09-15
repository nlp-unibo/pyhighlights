"""The released ``toy_dataset.pkl``, as this reproduction reads it."""

from cinnamon.configuration import Param
from cinnamon.registry import register_class

from pyhighlights.configurations.datasets import LoaderConfig
from pyhighlights_benchmarks.genspp2025.configurations.keys import (
    NAMESPACE,
)
from pyhighlights_benchmarks.genspp2025.corpora import GenSPPToyLoader


@register_class(
    name="dataset",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights_benchmarks.genspp2025.corpora.GenSPPToyLoader",
)
class ToyConfig(LoaderConfig):
    """The released ``toy_dataset.pkl``, not a regenerated corpus."""

    #: The published artifact and its digest, the same ones
    #: :class:`GenSPPToyLoader` defaults to. Left null, the registered key --
    #: the one the reproduction documents -- could not be built at all: the
    #: loader has nowhere to read the corpus from and says so.
    url: str | None = Param(GenSPPToyLoader.URL)
    sha256: str | None = Param(GenSPPToyLoader.SHA256)
    train_ratio: float = Param(0.8, gt=0.0, lt=1.0)
    val_ratio: float = Param(0.2, ge=0.0, lt=1.0)
    split_seed: int = Param(15000)
