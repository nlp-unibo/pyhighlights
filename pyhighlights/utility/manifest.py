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

What is deliberately absent: the task's own registration key. A component is
built as ``component_class(**{**config.values, **build_args})`` and is never
told which key produced it, so a task cannot record what it does not know. The
tree below is enough to rebuild the run -- it is every argument the task
received -- it just cannot be replayed as a single ``Registry.from_key`` call.
"""

from __future__ import annotations

import importlib.metadata
import platform
from typing import Any, Dict, Mapping, Sequence

from cinnamon.registry import RegistrationKey, Registry

__all__ = ["PACKAGES", "describe", "resolve", "versions"]

#: The packages whose version can change a number. Anything else installed
#: alongside them is noise in a file somebody has to read.
PACKAGES = ("pyhighlights", "cinnamon-core", "torch", "lightning")


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


def resolve(value: Any) -> Any:
    """Replace every registration key with the values behind it.

    Containers keep their shape, so a list of metric keys becomes a list of
    metric settings in the same order. A key that appears twice -- the same
    backbone under a selector and a predictor -- is written out twice, which
    reads better than a file of cross-references.
    """
    if isinstance(value, RegistrationKey):
        configuration = Registry.retrieve_configuration(registration_key=value)
        return {
            "key": str(value),
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
    """
    kind = type(component)
    return {
        "component": f"{kind.__module__}.{kind.__qualname__}",
        "versions": versions(),
        "settings": {
            name: resolve(value)
            for name, value in vars(component).items()
            if not name.startswith("_")
        },
    }
