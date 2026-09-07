from typing import Dict, List, Sequence

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.components.datasets import (
    R2A_TASKS,
    R2A_URL,
    ERASERLoader,
    HateXplainLoader,
)

NAMESPACE = "pyhighlights"


def key(name: str, *tags: str) -> RegistrationKey:
    return RegistrationKey(name=name, tags=set(tags), namespace=NAMESPACE)


R2A = key("dataset", "r2a")
HATEXPLAIN = key("dataset", "hatexplain")
ERASER = key("dataset", "eraser")
TOY = key("dataset", "toy")


class LoaderConfig(Configuration):
    """Fields every loader shares."""

    directory: str | None = Param(None)
    remove_leakage: bool = Param(True)
    key: str = Param("text")


class R2AConfig(LoaderConfig):
    """Beer and Hotel aspects; one variant per task."""

    # Cinnamon rejects a default that also appears in the variant list.
    task: str = Param(
        "hotel_Location",
        variants=[task for task in R2A_TASKS if task != "hotel_Location"],
    )
    splits: Dict[str, str] | None = Param(None)
    url: str = Param(R2A_URL)
    sha256: str | None = Param(None)

    @classmethod
    @register_method(
        name="dataset",
        tags={"r2a"},
        namespace=NAMESPACE,
        component="pyhighlights.components.datasets.R2ALoader",
    )
    def default(cls):
        return super().default()


class HateXplainConfig(LoaderConfig):
    """Hate-speech posts with aggregated labels and token rationales."""

    url: str = Param(HateXplainLoader.URL)
    divisions_url: str = Param(HateXplainLoader.DIVISIONS_URL)
    labels: Sequence[str] = Param(list(HateXplainLoader.LABELS))
    rationale: str = Param("majority", variants=["union", "intersection"])
    ties: str = Param("drop", variants=["keep"])

    @classmethod
    @register_method(
        name="dataset",
        tags={"hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.datasets.HateXplainLoader",
    )
    def default(cls):
        return super().default()


class ERASERConfig(LoaderConfig):
    """Single-document ERASER tasks; evidence spans become highlights."""

    task: str = Param("movies")
    splits: Dict[str, str] | None = Param(None)
    url: str = Param(ERASERLoader.URL)
    sha256: str | None = Param(None)

    @classmethod
    @register_method(
        name="dataset",
        tags={"eraser"},
        namespace=NAMESPACE,
        component="pyhighlights.components.datasets.ERASERLoader",
    )
    def default(cls):
        return super().default()


class ToyConfig(LoaderConfig):
    """Synthetic corpus for smoke tests and demos; no download."""

    sizes: Dict[str, int] | None = Param(None)
    triggers: Sequence[str] = Param(["a great film", "a dull film"])
    length: int = Param(24, ge=1)
    vocabulary_size: int = Param(32, ge=1)
    seed: int = Param(0)

    @classmethod
    @register_method(
        name="dataset",
        tags={"toy"},
        namespace=NAMESPACE,
        component="pyhighlights.components.datasets.ToyLoader",
    )
    def default(cls):
        return super().default()


__all__: List[str] = [
    "ERASER",
    "ERASERConfig",
    "HATEXPLAIN",
    "HateXplainConfig",
    "LoaderConfig",
    "R2A",
    "R2AConfig",
    "TOY",
    "ToyConfig",
]
