from typing import Dict

import torch as th

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.phased import PhasedSPP


class MCD(PhasedSPP):
    """Rationalizer trained against selected-input and full-input predictions.

    A predictor reading the full input guides the generator: a highlight that
    d-separates the label from the rest of the input makes the selected-input
    and full-input predictions agree.

    Liu, Wang, Wang, Li, Deng, Zhang and Qiu, 2023, *D-Separation for Causal
    Self-Explanation*, NeurIPS 2023.
    Reference implementation:
    <https://github.com/jugechengzi/Rationalization-MCD>.

    The phases and the three loss lists are
    :class:`~pyhighlights.components.models.spp.phased.PhasedSPP`'s. What MCD
    adds is the pass they are scored over: the predictor reads the highlight,
    and every namespace exposes ``full_class_logits`` beside the
    selected-input fields.
    """

    #: The paper's name for the phase that trains the predictor, kept so a
    #: training log reads as the published algorithm does.
    predictor_phase = "classifier"

    def phase_class_logits(
        self, input_data: InputData, highlight_mask: th.Tensor, selection: th.Tensor
    ) -> th.Tensor:
        """The highlight pass, which MCD trains on: in the graph, per phase."""
        return self.predict(input_data, selection)

    def extra_logits(
        self, input_data: InputData, selection: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """The full input, which is what the highlight has to agree with."""
        return {"full_class_logits": self.predict_full(input_data)}
