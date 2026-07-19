"""
Utilities for building torchmetrics metrics dynamically by name.
"""

from __future__ import annotations

import difflib
import importlib
import inspect
from typing import Any, Dict, Iterable, Optional, Tuple, List

import torchmetrics


def _resolve_metric_class(name: str) -> Tuple[Optional[type], str, Iterable[str]]:
    """
    Resolve `name` to (class_or_None, namespace_searched, candidate_names).

    - No dot -> looked up directly on `torchmetrics`
      (e.g. "F1Score" -> torchmetrics.F1Score).
    - Dotted -> everything before the last dot is a submodule path,
      relative to `torchmetrics` unless already qualified
      (e.g. "classification.F1Score" -> torchmetrics.classification.F1Score,
      "torchmetrics.classification.F1Score" works the same way).

    `candidate_names` is whatever namespace was actually searched, used
    for typo suggestions.
    """
    if "." not in name:
        return getattr(torchmetrics, name, None), "torchmetrics", dir(torchmetrics)

    module_path, _, class_name = name.rpartition(".")
    full_module_path = (
        module_path
        if module_path.startswith("torchmetrics")
        else f"torchmetrics.{module_path}"
    )

    try:
        module = importlib.import_module(full_module_path)
    except ImportError as e:
        raise ValueError(
            f"Could not import submodule '{full_module_path}' while resolving '{name}': {e}"
        ) from e

    return getattr(module, class_name, None), full_module_path, dir(module)


def build_torchmetric(name: str, **kwargs: Any) -> torchmetrics.Metric:
    """
    Instantiate a torchmetrics metric by name.

    Parameters
    ----------
    name:
        Either a bare class name resolved on `torchmetrics` directly
        (e.g. "F1Score", "Accuracy"), or a dotted path resolved as
        `<submodule>.<ClassName>` relative to `torchmetrics`
        (e.g. "classification.F1Score", "regression.MeanSquaredError").
        Case-sensitive.
    **kwargs:
        Forwarded to the metric's constructor
        (e.g. task="multiclass", num_classes=5).

    Returns
    -------
    An instantiated `torchmetrics.Metric`.

    Raises
    ------
    ValueError:
        If `name`'s submodule can't be imported, or doesn't resolve to a
        torchmetrics.Metric subclass.
    TypeError:
        If `kwargs` don't match the metric's constructor signature.

    Examples
    --------
    >>> build_torchmetric("F1Score", task="multiclass", num_classes=5)
    >>> build_torchmetric("classification.F1Score", task="multiclass", num_classes=5)
    >>> build_torchmetric("Accuracy", task="binary")
    """
    metric_cls, namespace, candidates = _resolve_metric_class(name)
    class_name = name.rpartition(".")[2] if "." in name else name

    if not (
        inspect.isclass(metric_cls) and issubclass(metric_cls, torchmetrics.Metric)
    ):
        suggestion = _closest_match(class_name, candidates)
        hint = f" Did you mean '{suggestion}' in {namespace}?" if suggestion else ""
        raise ValueError(
            f"'{name}' does not resolve to a torchmetrics.Metric class in {namespace}.{hint}"
        )

    try:
        return metric_cls(**kwargs)
    except TypeError as e:
        raise TypeError(
            f"Failed to build {namespace}.{class_name}(**{kwargs}): {e}"
        ) from e


def build_torchmetrics(
    specs: Dict[str, Dict[str, Any]],
) -> torchmetrics.MetricCollection:
    """
    Build several metrics at once from a {name: kwargs} mapping and wrap
    them in a MetricCollection (convenient for logging/updating them
    together, e.g. inside a Lightning module).

    Examples
    --------
    >>> build_torchmetrics({
    ...     "f1": {"name": "F1Score", "task": "multiclass", "num_classes": 5},
    ...     "acc": {"name": "Accuracy", "task": "multiclass", "num_classes": 5},
    ... })
    """
    return torchmetrics.MetricCollection(
        {name: build_torchmetric(**kwargs) for name, kwargs in specs.items()}
    )


def _closest_match(name: str, candidates: Iterable[str]) -> str | None:
    matches = difflib.get_close_matches(name, candidates, n=1, cutoff=0.6)
    return matches[0] if matches else None
