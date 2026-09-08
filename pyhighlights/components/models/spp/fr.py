from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.spp.base import SPP, SPPBackbone


class FR(SPP):
    """Folded rationalization with one selector and one shared backbone.

    Selector and predictor fold onto the same encoder, so the predictor's
    gradient reaches the selector through shared weights and the two modules
    cannot drift apart the way an independent pair does.

    Liu, Wang, Wang, Li, Yue and Zhang, 2022, *FR: Folded Rationalization with
    a Unified Encoder*, NeurIPS 2022.
    Reference implementation: <https://github.com/jugechengzi/FR>.
    """

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
