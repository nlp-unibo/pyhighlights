"""Optimizer registrations."""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_method

from pyhighlights.configurations.keys import NAMESPACE


class AdamConfig(Configuration):
    lr: float = Param(1e-3, gt=0.0)
    weight_decay: float = Param(0.0, ge=0.0)

    @classmethod
    @register_method(
        name="optimizer",
        tags={"adam"},
        namespace=NAMESPACE,
        component="torch.optim.Adam",
    )
    def default(cls):
        return super().default()
