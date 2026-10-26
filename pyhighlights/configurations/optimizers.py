"""Optimizer registrations."""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_class

from pyhighlights.configurations.keys import ADAM


@register_class(key=ADAM, component="torch.optim.Adam")
class AdamConfig(Configuration):
    lr: float = Param(1e-3, gt=0.0)
    weight_decay: float = Param(0.0, ge=0.0)
