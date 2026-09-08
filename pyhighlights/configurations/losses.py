"""Criterion registrations and the bindings that feed them named fields."""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey, register_method

from pyhighlights.configurations.keys import (
    CONTIGUITY_PENALTY,
    CROSS_ENTROPY,
    JS_DIV,
    KL_DIV,
    MASKED_BCE,
    MASKED_CROSS_ENTROPY,
    NAMESPACE,
    SPARSITY_PENALTY,
)


class CrossEntropyConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"cross_entropy"},
        namespace=NAMESPACE,
        component="torch.nn.CrossEntropyLoss",
    )
    def default(cls):
        return super().default()


class MaskedCrossEntropyConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"masked_cross_entropy"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.MaskedCrossEntropy",
    )
    def default(cls):
        return super().default()


class MaskedBCEConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"masked_bce"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.MaskedBinaryCrossEntropy",
    )
    def default(cls):
        return super().default()


class KLDivConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"kl_div"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.KLDiv",
    )
    def default(cls):
        return super().default()


class JSDivConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"js_div"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.JSDiv",
    )
    def default(cls):
        return super().default()


class ContiguityPenaltyConfig(Configuration):
    @classmethod
    @register_method(
        name="criterion",
        tags={"contiguity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.ContiguityPenalty",
    )
    def default(cls):
        return super().default()


class SparsityPenaltyConfig(Configuration):
    threshold: float = Param(0.15, ge=0.0, le=1.0)

    @classmethod
    @register_method(
        name="criterion",
        tags={"sparsity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.SparsityPenalty",
    )
    def default(cls):
        return super().default()


class LossConfig(Configuration):
    """Binds a criterion to the namespace fields it scores."""

    name: str = Param("classification")
    loss: RegistrationKey[th.nn.Module] = Param(CROSS_ENTROPY)
    inputs: List[str] = Param(["class_logits", "y_true"])
    coefficient: float = Param(1.0, ge=0.0)
    enabled: bool = Param(True)

    @classmethod
    @register_method(
        name="loss",
        tags={"classification"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class FullClassificationLossConfig(LossConfig):
    name: str = Param("full_classification")
    inputs: List[str] = Param(["full_class_logits", "y_true"])

    @classmethod
    @register_method(
        name="loss",
        tags={"classification", "full"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class HighlightLossConfig(LossConfig):
    name: str = Param("highlight")
    loss: RegistrationKey[th.nn.Module] = Param(MASKED_CROSS_ENTROPY)
    inputs: List[str] = Param(["highlight_logits", "highlight_true", "mask"])

    @classmethod
    @register_method(
        name="loss",
        tags={"highlight"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class SparsityLossConfig(LossConfig):
    name: str = Param("sparsity")
    loss: RegistrationKey[th.nn.Module] = Param(SPARSITY_PENALTY)
    inputs: List[str] = Param(["highlight_mask", "mask"])

    @classmethod
    @register_method(
        name="loss",
        tags={"sparsity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class ContiguityLossConfig(LossConfig):
    name: str = Param("contiguity")
    loss: RegistrationKey[th.nn.Module] = Param(CONTIGUITY_PENALTY)
    inputs: List[str] = Param(["highlight_mask", "mask"])
    coefficient: float = Param(2.0, ge=0.0)

    @classmethod
    @register_method(
        name="loss",
        tags={"contiguity"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class DiscrepancyLossConfig(LossConfig):
    name: str = Param("discrepancy")
    loss: RegistrationKey[th.nn.Module] = Param(KL_DIV)
    inputs: List[str] = Param(["class_logits", "full_class_logits"])

    @classmethod
    @register_method(
        name="loss",
        tags={"discrepancy"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class GuideLossConfig(LossConfig):
    name: str = Param("guide")
    loss: RegistrationKey[th.nn.Module] = Param(MASKED_BCE)
    inputs: List[str] = Param(["selection_logits", "guide_target", "mask"])

    @classmethod
    @register_method(
        name="loss",
        tags={"guide"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()


class JSDLossConfig(LossConfig):
    name: str = Param("jsd")
    loss: RegistrationKey[th.nn.Module] = Param(JS_DIV)
    inputs: List[str] = Param(["class_logits", "guider_class_logits"])

    @classmethod
    @register_method(
        name="loss",
        tags={"jsd"},
        namespace=NAMESPACE,
        component="pyhighlights.utility.losses.Loss",
    )
    def default(cls):
        return super().default()
