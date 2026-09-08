"""Optimizer registrations."""

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import register_method

#: Namespace cinnamon resolves this module's registrations under. Kept a literal
#: in every registering module: ``NamespaceExtractor`` reads it statically and only
#: sees bindings made in the same file.
NAMESPACE = "pyhighlights"


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


class GenSPPAdamConfig(AdamConfig):
    lr: float = Param(1e-2, gt=0.0)

    @classmethod
    @register_method(
        name="optimizer",
        tags={"adam", "genspp"},
        namespace=NAMESPACE,
        component="torch.optim.Adam",
    )
    def default(cls):
        return super().default()
