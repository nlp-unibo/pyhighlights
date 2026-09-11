"""What a run was: the settings it was given, and the versions that ran it.

A results directory is worth what can be rebuilt from it. Metrics say what
happened; they do not say what produced them, and a task's own attributes are
not enough either -- a task holds *keys*, so ``model`` reads
``name=model--tags=['fr','gru']`` and the hidden size, the sparsity threshold
and the learning rate behind that key are nowhere in the record.

:func:`describe` writes down the whole tree instead. Every key is replaced by
the configuration it names, recursively, so a manifest states the numbers the
run actually used rather than the names of the places they came from. Alongside
them go the versions of the packages that did the computing, since a metric
that moved between two runs of the same configuration is a version difference
or nothing at all.

The record also names the key that built the component and the arguments the
caller overrode, which is what makes a run replayable rather than merely
readable: the key alone rebuilds the registered defaults, not the run that was
launched. Both come from cinnamon 2.0.3, which annotates every component it
builds; a component built by hand has neither, and says so with ``null``.
"""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Any, Dict, Mapping, Sequence

from cinnamon.registry import RegistrationKey, Registry

__all__ = ["ANNOTATIONS", "PACKAGES", "describe", "resolve", "versions"]

#: The packages whose version can change a number. Anything else installed
#: alongside them is noise in a file somebody has to read.
PACKAGES = ("pyhighlights", "cinnamon-core", "torch", "lightning")

#: What cinnamon puts on a component it builds. Reported as a record of its
#: own rather than left among the settings, where it would read as something
#: the task was configured with.
ANNOTATIONS = ("registration_key", "build_args")


def versions() -> Dict[str, str]:
    """The interpreter and the packages that do the computing.

    A package that is not installed is left out rather than reported as
    ``None``: transformers is optional, and a run that never imported it is
    not a run whose transformers version was unknown.
    """
    found = {"python": platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            continue
    return found


#: Where a resolved entry records the registration key it came from.
#:
#: Not ``"key"``. A resolved entry is the registration key beside the
#: configuration's own parameters, flattened into one object, so any name a
#: parameter can take is a name the key can lose -- and
#: :class:`~pyhighlights.components.preprocessors.LeakageRemover` takes exactly
#: this one, to name the column it deduplicates on. Its entry used to read
#: ``"key": "text"`` with the registration key gone, which
#: :meth:`~pyhighlights.components.analyzers.PredictionAnalyzer.corpus` then
#: tried to parse as a key. ``@key`` is not a Python identifier, so no
#: parameter can ever be called it.
KEY_FIELD = "@key"


def registration_key(entry: Mapping[str, Any]) -> str:
    """The registration key of a resolved entry, old manifests included.

    Manifests written before :data:`KEY_FIELD` existed record it as ``key``,
    and a results tree outlives the release that wrote it.
    """
    if KEY_FIELD in entry:
        return entry[KEY_FIELD]
    return entry["key"]


def resolve(value: Any) -> Any:
    """Replace every registration key with the values behind it.

    Containers keep their shape, so a list of metric keys becomes a list of
    metric settings in the same order. A key that appears twice -- the same
    backbone under a selector and a predictor -- is written out twice, which
    reads better than a file of cross-references.

    The key itself is recorded under :data:`KEY_FIELD` rather than ``key``, so
    that a configuration with a parameter of that name keeps both.
    """
    if isinstance(value, RegistrationKey):
        configuration = Registry.retrieve_configuration(registration_key=value)
        return {
            KEY_FIELD: str(value),
            **{name: resolve(item) for name, item in configuration.values.items()},
        }
    if isinstance(value, Mapping):
        return {name: resolve(item) for name, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [resolve(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return sorted(resolve(item) for item in value)
    return value


def describe(component: Any) -> Dict[str, Any]:
    """The manifest for one built component.

    Private attributes are what the run *built* rather than what it was asked
    for -- an embedding matrix fitted against the training split is among them
    -- and they are no more a setting than the trained weights are.

    ``key`` and ``build_args`` are what cinnamon wrote on the component when it
    built it, and are ``null`` for a component nobody built through a registry.
    """
    kind = type(component)
    key = getattr(component, "registration_key", None)
    build_args = getattr(component, "build_args", None)
    return {
        "component": f"{kind.__module__}.{kind.__qualname__}",
        "key": None if key is None else str(key),
        "build_args": None if build_args is None else resolve(build_args),
        "versions": versions(),
        "settings": {
            name: resolve(value)
            for name, value in vars(component).items()
            if not name.startswith("_") and name not in ANNOTATIONS
        },
    }
