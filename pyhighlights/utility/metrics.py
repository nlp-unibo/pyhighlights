from __future__ import annotations

from typing import Dict

from cinnamon.registry import RegistrationKey, Registry
from torchmetrics import Metric, MetricCollection


def build_torchmetric(key: RegistrationKey[Metric]) -> Metric:
    return Registry.from_key(key, expected_type=Metric)


def build_torchmetrics(keys: Dict[str, RegistrationKey[Metric]]) -> MetricCollection:
    return MetricCollection(Registry.from_keys(keys, expected_type=Metric))
