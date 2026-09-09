"""Corpus loader registrations."""

from typing import Dict, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_method

from pyhighlights.components.loaders import (
    BEER_TASKS,
    HOTEL_TASKS,
    R2A_URL,
    ERASERLoader,
    HateXplainLoader,
)
from pyhighlights.configurations.keys import NAMESPACE


class LoaderConfig(Configuration):
    """Fields every loader shares."""

    directory: str | None = Param(None)


class R2AConfig(LoaderConfig):
    """Fields the Beer and Hotel loaders share; the archive holds both."""

    splits: Dict[str, str] | None = Param(None)
    url: str = Param(R2A_URL)
    sha256: str | None = Param(None)


class BeerConfig(R2AConfig):
    """Beer aspects; one variant per aspect."""

    # Cinnamon rejects a default that also appears in the variant list.
    task: str = Param(BEER_TASKS[0], variants=list(BEER_TASKS[1:]))

    @classmethod
    @register_method(
        name="dataset",
        tags={"beer"},
        namespace=NAMESPACE,
        component="pyhighlights.components.loaders.BeerLoader",
    )
    def default(cls):
        return super().default()


class HotelConfig(R2AConfig):
    """Hotel aspects; one variant per aspect."""

    task: str = Param(HOTEL_TASKS[0], variants=list(HOTEL_TASKS[1:]))

    @classmethod
    @register_method(
        name="dataset",
        tags={"hotel"},
        namespace=NAMESPACE,
        component="pyhighlights.components.loaders.HotelLoader",
    )
    def default(cls):
        return super().default()


class HateXplainConfig(LoaderConfig):
    """Hate-speech posts, every annotator judgement kept."""

    url: str = Param(HateXplainLoader.URL)
    divisions_url: str = Param(HateXplainLoader.DIVISIONS_URL)

    @classmethod
    @register_method(
        name="dataset",
        tags={"hatexplain"},
        namespace=NAMESPACE,
        component="pyhighlights.components.loaders.HateXplainLoader",
    )
    def default(cls):
        return super().default()


class MoviesConfig(LoaderConfig):
    """ERASER movies: evidence spans become highlights."""

    task: str = Param("movies")
    splits: Dict[str, str] | None = Param(None)
    url: str = Param(ERASERLoader.URL)
    sha256: str | None = Param(None)

    @classmethod
    @register_method(
        name="dataset",
        tags={"movies"},
        namespace=NAMESPACE,
        component="pyhighlights.components.loaders.MoviesLoader",
    )
    def default(cls):
        return super().default()


class ToyConfig(LoaderConfig):
    """Synthetic corpus for smoke tests and demos; no download."""

    sizes: Dict[str, int] | None = Param(None)
    triggers: List[str] = Param(["a great film", "a dull film"])
    length: int = Param(24, ge=1)
    vocabulary_size: int = Param(32, ge=1)
    seed: int = Param(0)

    @classmethod
    @register_method(
        name="dataset",
        tags={"toy"},
        namespace=NAMESPACE,
        component="pyhighlights.components.loaders.ToyLoader",
    )
    def default(cls):
        return super().default()


__all__: List[str] = [
    "BeerConfig",
    "HateXplainConfig",
    "HotelConfig",
    "LoaderConfig",
    "MoviesConfig",
    "R2AConfig",
    "ToyConfig",
]
