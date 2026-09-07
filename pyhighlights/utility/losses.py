from __future__ import annotations

import abc
import math
from typing import List

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.data import InputData, OutputData, SPPOutput


class Loss(th.nn.Module, abc.ABC):
    def __init__(
        self,
        name: str,
        coefficient: float = 1.0,
        enabled: bool = True,
    ):
        super().__init__()
        self.name = name
        self.coefficient = coefficient
        self.enabled = enabled

    @abc.abstractmethod
    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor: ...


def build_loss(key: RegistrationKey[Loss]) -> Loss:
    return Registry.from_key(key, expected_type=Loss)


def build_losses(keys: List[RegistrationKey[Loss]]) -> List[Loss]:
    return Registry.from_keys(keys, expected_type=Loss)


class LossWrapper(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.Module],
        loss_input: str,
        loss_target: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss, expected_type=th.nn.Module)
        self.loss_input = loss_input
        self.loss_target = loss_target

    @staticmethod
    def _tensor_attribute(
        name: str, input_data: InputData, output_data: OutputData
    ) -> th.Tensor:
        for data in (input_data, output_data):
            value = getattr(data, name, None)
            if isinstance(value, th.Tensor):
                return value
        raise AttributeError(f"Could not find tensor attribute {name}")

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        return self.loss(
            self._tensor_attribute(self.loss_input, input_data, output_data),
            self._tensor_attribute(self.loss_target, input_data, output_data),
        )


class ClassificationLoss(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.CrossEntropyLoss],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss, expected_type=th.nn.CrossEntropyLoss)

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        return self.loss(input=output_data.class_logits, target=input_data.y_true)


class HighlightLoss(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.CrossEntropyLoss],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss, expected_type=th.nn.CrossEntropyLoss)

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        if not isinstance(output_data, SPPOutput):
            raise TypeError("HighlightLoss requires SPPOutput")

        logits = output_data.highlight_logits.reshape(-1, 2)
        targets = input_data.highlight_true.reshape(-1)
        valid = (targets != -1) & input_data.mask.reshape(-1).bool()
        if not valid.any():
            return logits.sum() * 0
        return self.loss(input=logits[valid], target=targets[valid].long())


class HighlightSparsityLoss(Loss):
    def __init__(self, sparsity_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)
        self.sparsity_threshold = sparsity_threshold

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        if not isinstance(output_data, SPPOutput):
            raise TypeError("HighlightSparsityLoss requires SPPOutput")
        denominator = input_data.mask.sum().clamp_min(1)
        sparsity = output_data.highlight_mask.sum() / denominator
        return th.abs(sparsity - self.sparsity_threshold)


class HighlightContiguityLoss(Loss):
    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        if not isinstance(output_data, SPPOutput):
            raise TypeError("HighlightContiguityLoss requires SPPOutput")
        if output_data.highlight_mask.shape[1] < 2:
            return output_data.highlight_mask.sum() * 0

        valid_pairs = input_data.mask[:, 1:].bool() & input_data.mask[:, :-1].bool()
        differences = th.abs(
            output_data.highlight_mask[:, 1:] - output_data.highlight_mask[:, :-1]
        )
        if not valid_pairs.any():
            return differences.sum() * 0
        return differences[valid_pairs].mean()


class KLDiv(th.nn.Module):
    def forward(self, p: th.Tensor, q: th.Tensor) -> th.Tensor:
        return th.nn.functional.kl_div(
            th.nn.functional.log_softmax(p, dim=-1),
            th.nn.functional.softmax(q, dim=-1),
            reduction="batchmean",
        )


class JSDiv(th.nn.Module):
    def forward(self, p: th.Tensor, q: th.Tensor) -> th.Tensor:
        log_p = th.nn.functional.log_softmax(p, dim=-1)
        log_q = th.nn.functional.log_softmax(q, dim=-1)
        log_mean = th.logaddexp(log_p, log_q) - math.log(2)
        return (
            th.nn.functional.kl_div(
                log_mean, log_p, reduction="batchmean", log_target=True
            )
            + th.nn.functional.kl_div(
                log_mean, log_q, reduction="batchmean", log_target=True
            )
        ) / 2
