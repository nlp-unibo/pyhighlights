"""Corpus loader registrations."""

from typing import Dict, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.components.loaders import (
    BEER_TASKS,
    HOTEL_TASKS,
    R2A_SHA256,
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
    #: The digest the loader defaults to, named again here because a
    #: registered key is what a run actually builds: leaving it null handed
    #: every registered run an unverified download while the documentation
    #: said the opposite. A fixture archive passes ``sha256=None`` explicitly.
    sha256: str | None = Param(R2A_SHA256)


@register_class(
    name="dataset",
    tags={"beer"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.BeerLoader",
)
class BeerConfig(R2AConfig):
    """Beer aspects; one variant per aspect."""

    # Cinnamon rejects a default that also appears in the variant list.
    task: str = Param(BEER_TASKS[0], variants=list(BEER_TASKS[1:]))


@register_class(
    name="dataset",
    tags={"hotel"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.HotelLoader",
)
class HotelConfig(R2AConfig):
    """Hotel aspects; one variant per aspect."""

    task: str = Param(HOTEL_TASKS[0], variants=list(HOTEL_TASKS[1:]))


@register_class(
    name="dataset",
    tags={"hatexplain"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.HateXplainLoader",
)
class HateXplainConfig(LoaderConfig):
    """Hate-speech posts, every annotator judgement kept."""

    url: str = Param(HateXplainLoader.URL)
    divisions_url: str = Param(HateXplainLoader.DIVISIONS_URL)
    #: As on :class:`R2AConfig`: the loader's defaults, named again where the
    #: registered key can be read off them. Two digests because they are two
    #: downloads.
    sha256: str | None = Param(HateXplainLoader.SHA256)
    divisions_sha256: str | None = Param(HateXplainLoader.DIVISIONS_SHA256)


@register_class(
    name="dataset",
    tags={"movies"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.MoviesLoader",
)
class MoviesConfig(LoaderConfig):
    """ERASER movies: evidence spans become highlights."""

    task: str = Param("movies")
    splits: Dict[str, str] | None = Param(None)
    url: str = Param(ERASERLoader.URL)
    #: As on :class:`R2AConfig`: the loader's default, said again where the
    #: registered key can be read off it.
    sha256: str | None = Param(ERASERLoader.SHA256)


@register_class(
    name="dataset",
    tags={"toy"},
    namespace=NAMESPACE,
    component="pyhighlights.components.loaders.ToyLoader",
)
class ToyConfig(LoaderConfig):
    """Synthetic corpus for smoke tests and demos; no download."""

    sizes: Dict[str, int] | None = Param(None)
    #: What each class is: a pattern, or a list of patterns all of which have
    #: to appear. Tokens are characters here, as they are in every toy corpus
    #: of this line of work. The default is the cheap smoke-test corpus, one
    #: pattern per class; a conjunction whose patterns are shared between
    #: classes is the one no single n-gram can solve.
    triggers: List[str | List[str]] = Param(["aa", "bcd"])
    length: int = Param(20, ge=1)
    #: How many filler characters, drawn from the letters no trigger uses.
    vocabulary_size: int = Param(20, ge=1)
    seed: int = Param(0)


__all__: List[str] = [
    "BeerConfig",
    "HateXplainConfig",
    "HotelConfig",
    "LoaderConfig",
    "MoviesConfig",
    "R2AConfig",
    "ToyConfig",
]
