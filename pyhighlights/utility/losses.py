import abc
from typing import List

import torch as th
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models.base import InputData, OutputData


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
    return Registry.from_key(key)


def build_losses(
    keys: List[RegistrationKey[Loss]],
) -> List[Loss]:
    return [build_loss(key) for key in keys]


class LossWrapper(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.Module],
        loss_input: str,
        loss_target: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss)
        self.loss_input = loss_input
        self.loss_target = loss_target

    def _check_tensor_attribute(
        self, name: str, input_data: InputData, output_data: OutputData
    ) -> th.Tensor:
        if hasattr(input_data, name):
            return getattr(input_data, name)
        elif hasattr(output_data, name):
            return getattr(output_data, name)
        else:
            raise AttributeError(f"Could not find attribute {name}")

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        loss_input = self._check_tensor_attribute(
            name=self.loss_input, input_data=input_data, output_data=output_data
        )
        loss_target = self._check_tensor_attribute(
            name=self.loss_target, input_data=input_data, output_data=output_data
        )

        return self.loss(loss_input, loss_target)


class ClassificationLoss(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.CrossEntropyLoss],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss)

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        # output_data.y_pred:       [bs, C]
        # input_data.y_true:        [bs,]

        return self.loss(input=output_data.y_pred, target=input_data.y_true)


class HighlightLoss(Loss):
    def __init__(
        self,
        loss: RegistrationKey[th.nn.CrossEntropyLoss],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.loss = Registry.from_key(loss)

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        # output_data.highlight_logits:     [bs, F, 2]
        # input_data.highlight_true:        [bs, F]

        return self.loss(
            input=output_data.highlight_logits.view(-1, 1),
            target=input_data.highlight_true.view(
                -1,
            ),
        )


class HighlightSparsityLoss(Loss):
    def __init__(self, sparsity_threshold: float = 0.15, **kwargs):
        super().__init__(**kwargs)

        self.sparsity_threshold = sparsity_threshold

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        # output_data.highlight_pred:   [bs, F]
        # input_data.mask:              [bs, F]

        # [bs,]
        sparsity = th.sum(output_data.highlight_pred) / th.sum(input_data.mask)
        return th.abs(sparsity - self.sparsity_threshold)


class HighlightContiguityLoss(Loss):
    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        # output_data.highlight_pred:   [bs, F]

        return th.mean(
            th.abs(
                output_data.highlight_pred[:, 1:] - output_data.highlight_pred[:, :-1]
            )
        )


# TODO: update
class JSDivLoss(Loss):
    def __init__(self, loss: RegistrationKey[th.nn.KLDivLoss], **kwargs):
        super(JSDivLoss, self).__init__(**kwargs)

        self.loss = Registry.from_key(loss)

    def forward(self, input_data: InputData, output_data: OutputData) -> th.Tensor:
        loss_input = self._check_tensor_attribute(
            name=self.input, input_data=input_data, output_data=output_data
        )
        loss_target = self._check_tensor_attribute(
            name=self.output, input_data=input_data, output_data=output_data
        )

        p_s = th.nn.functional.softmax(loss_input, dim=-1)
        q_s = th.nn.functional.softmax(loss_target, dim=-1)
        p_s, q_s = p_s.view(-1, p_s.size(-1)), q_s.view(-1, q_s.size(-1))
        m = (0.5 * (p_s + q_s)).log()
        return 0.5 * (self.loss(m, p_s.log()) + self.loss(m, q_s.log()))


# TODO: update
class KL_Div(th.nn.Module):
    def forward(self, p, q):
        return th.nn.functional.kl_div(
            th.nn.functional.softmax(p, dim=-1).log(),
            th.nn.functional.softmax(q, dim=-1),
            reduction="batchmean",
        )
