"""
Utilities for building torchmetrics metrics dynamically by name.
"""

from __future__ import annotations

from typing import Dict

from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric, MetricCollection

# TODO: define general interface just like Loss

def build_torchmetric(key: RegistrationKey[Metric]) -> Metric:
    return Registry.from_key(key)


def build_torchmetrics(
    keys: Dict[str, RegistrationKey[Metric]],
) -> MetricCollection:
    return MetricCollection(
        {name: build_torchmetric(key=key) for name, key in keys.items()}
    )
