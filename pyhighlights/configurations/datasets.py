"""Corpus loader registrations."""

from typing import Dict, List

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import (
    BEER,
    HATEXPLAIN,
    HOTEL,
    MOVIES,
    TOY,
)
from pyhighlights.constants import (
    BEER_TASKS,
    ERASER_SHA256,
    ERASER_URL,
    HATEXPLAIN_DIVISIONS_SHA256,
    HATEXPLAIN_DIVISIONS_URL,
    HATEXPLAIN_SHA256,
    HATEXPLAIN_URL,
    HOTEL_TASKS,
    R2A_SHA256,
    R2A_URL,
)


class LoaderConfig(Configuration):
    """Fields every loader shares."""

    directory: str | None = Param(
        None,
        description=(
            "Directory the corpus is downloaded to and read from. Unset means the "
            "shared cache, which ``PYHIGHLIGHTS_CACHE`` overrides."
        ),
    )


class R2AConfig(LoaderConfig):
    """Fields the Beer and Hotel loaders share; the archive holds both."""

    splits: Dict[str, str] | None = Param(
        None,
        description=(
            "File pattern of each split under the archive's ``data`` directory, "
            "keyed by split name, where ``{task}`` stands for the aspect. Unset "
            "reads the default train, val and test files."
        ),
    )
    url: str = Param(
        R2A_URL,
        description="URL of the R2A archive, which holds both Beer and Hotel.",
    )
    sha256: str | None = Param(
        R2A_SHA256,
        description=(
            "SHA-256 digest the downloaded archive is verified against, so a "
            "changed upstream fails loudly; a fixture archive passes ``None`` to "
            "skip the check. The loader's default is named again here because a "
            "registered key is what a run actually builds."
        ),
    )


@register_class(key=BEER, component="pyhighlights.components.loaders.BeerLoader")
class BeerConfig(R2AConfig):
    """Beer aspects; one variant per aspect."""

    task: str = Param(
        BEER_TASKS[0],
        variants=list(BEER_TASKS[1:]),
        description=(
            "Beer aspect to load. The first aspect is the default and the others "
            "are variants, because Cinnamon rejects a default that also appears "
            "in the variant list."
        ),
    )


@register_class(key=HOTEL, component="pyhighlights.components.loaders.HotelLoader")
class HotelConfig(R2AConfig):
    """Hotel aspects; one variant per aspect."""

    task: str = Param(
        HOTEL_TASKS[0],
        variants=list(HOTEL_TASKS[1:]),
        description=(
            "Hotel aspect to load: location, service or cleanliness, one variant "
            "per aspect."
        ),
    )


@register_class(
    key=HATEXPLAIN, component="pyhighlights.components.loaders.HateXplainLoader"
)
class HateXplainConfig(LoaderConfig):
    """Hate-speech posts, every annotator judgement kept."""

    url: str = Param(
        HATEXPLAIN_URL,
        description="URL of the HateXplain posts file, pinned to one upstream commit.",
    )
    divisions_url: str = Param(
        HATEXPLAIN_DIVISIONS_URL,
        description=(
            "URL of the official split map, pinned to the same commit as the posts."
        ),
    )
    sha256: str | None = Param(
        HATEXPLAIN_SHA256,
        description=(
            "SHA-256 digest the posts file is verified against, or ``None`` to "
            "skip the check. It repeats the loader's default so the registered "
            "key can be read off it."
        ),
    )
    divisions_sha256: str | None = Param(
        HATEXPLAIN_DIVISIONS_SHA256,
        description=(
            "SHA-256 digest the split map is verified against, or ``None`` to skip "
            "the check. It is separate from ``sha256`` because the two files are "
            "two downloads."
        ),
    )


@register_class(key=MOVIES, component="pyhighlights.components.loaders.MoviesLoader")
class MoviesConfig(LoaderConfig):
    """ERASER movies: evidence spans become highlights."""

    task: str = Param(
        "movies",
        description=(
            "ERASER task to load; ``movies`` is the only single-document task, "
            "and so the only one supported."
        ),
    )
    splits: Dict[str, str] | None = Param(
        None,
        description=(
            "JSONL file of each split inside the task directory, keyed by split "
            "name. Unset reads ``train.jsonl``, ``val.jsonl`` and ``test.jsonl``."
        ),
    )
    url: str = Param(
        ERASER_URL,
        description=(
            "URL template of the ERASER archive, where ``{task}`` stands for the "
            "task name."
        ),
    )
    sha256: str | None = Param(
        ERASER_SHA256,
        description=(
            "SHA-256 digest the downloaded archive is verified against, or "
            "``None`` to skip the check. It repeats the loader's default so the "
            "registered key can be read off it."
        ),
    )


@register_class(key=TOY, component="pyhighlights.components.loaders.ToyLoader")
class ToyConfig(LoaderConfig):
    """Synthetic corpus for sanity check runs and demos; no download."""

    sizes: Dict[str, int] | None = Param(
        None,
        description=(
            "Number of examples generated for each split, keyed by split name. "
            "Unset generates 64 train, 16 val and 16 test examples."
        ),
    )
    triggers: List[str | List[str]] = Param(
        ["aa", "bc"],
        description=(
            "One trigger per class: a pattern of characters, or a list of "
            "patterns that all have to appear. The default is one pattern per "
            "class for a cheap sanity check, while patterns shared between "
            "classes make a corpus no single n-gram can solve."
        ),
    )
    length: int = Param(
        20,
        ge=1,
        description=(
            "Number of tokens in every document, patterns included; tokens are "
            "characters, and the length is at least 1."
        ),
    )
    contaminations: int = Param(
        0,
        ge=0,
        description=(
            "Number of pattern chunks scattered through the filler of each "
            "example, without which a fragment of a pattern classifies as well as "
            "the pattern. It is 0 here because the registered corpus is for a "
            "cheap sanity check run."
        ),
    )
    min_chunk: int = Param(
        2,
        ge=1,
        description=(
            "Shortest piece of a pattern, in characters, worth cutting out as a "
            "chunk; at least 1."
        ),
    )
    vocabulary_size: int = Param(
        20,
        ge=1,
        description=(
            "Number of filler characters, drawn from the letters no trigger uses; "
            "at least 1."
        ),
    )
    seed: int = Param(
        0,
        description=(
            "Random seed of the generator, combined with each split name so every "
            "split draws its own examples."
        ),
    )


__all__: List[str] = [
    "BeerConfig",
    "HateXplainConfig",
    "HotelConfig",
    "LoaderConfig",
    "MoviesConfig",
    "R2AConfig",
    "ToyConfig",
]
