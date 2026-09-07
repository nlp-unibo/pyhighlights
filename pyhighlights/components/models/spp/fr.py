from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.spp.base import SPP, SPPBackbone


class FR(SPP):
    """Folded rationalization with one selector and one shared backbone."""

    def __init__(
        self,
        predictor_backbone: RegistrationKey[SPPBackbone] | None = None,
        **kwargs,
    ):
        if predictor_backbone is not None:
            raise ValueError("FR requires a shared selector/predictor backbone")
        super().__init__(predictor_backbone=None, **kwargs)
        if len(self.selectors) != 1:
            raise ValueError("FR requires exactly one selector")
