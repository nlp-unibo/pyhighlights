from typing import Dict

import torch as th

from pyhighlights.components.models.base import InputData
from pyhighlights.components.models.spp.phased import PhasedSPP


class MRD(PhasedSPP):
    """Rationalizer trained by what is left once the highlight is removed.

    Maximum mutual information asks the highlight to predict the label, which
    a spurious feature correlated with the label answers just as well. MRD
    asks the opposite question: remove the highlight, and what remains should
    stop looking like the whole input. Removing plain noise or a spurious
    feature leaves the conditional distribution of the rest unchanged, so only
    the causal features move it -- which makes a corpus full of spurious
    features behave like a clean one rather than needing a penalty per
    spurious feature.

    Two consequences that make this model read oddly beside the others. The
    predictor never trains on the highlight: it is trained on the
    **complement** and on the full input, and the highlight pass exists only
    so the metrics have something to score. And the generator maximizes a
    divergence rather than minimizing one, which is a loss with a negative
    coefficient.

    Liu, Deng, Niu, Wang, Wang, Zhang and Li, 2024, *Is the MMI Criterion
    Necessary for Interpretability? Degenerating Non-causal Features to Plain
    Noise for Self-Rationalization*, NeurIPS 2024.
    Paper: <https://proceedings.neurips.cc/paper_files/paper/2024/hash/d53d51e88d92d3723755f6d425bc513b-Abstract-Conference.html>.
    Reference implementation:
    <https://github.com/jugechengzi/Rationalization-MRD>.

    The phases and the three loss lists are
    :class:`~pyhighlights.components.models.spp.phased.PhasedSPP`'s, as MCD's
    are. Every namespace carries ``complement_class_logits`` and
    ``full_class_logits`` beside the highlight's own fields.
    """

    def phase_class_logits(
        self, input_data: InputData, highlight_mask: th.Tensor, selection: th.Tensor
    ) -> th.Tensor:
        """The highlight pass, outside the graph.

        Nothing trains on it, and the reference implementation keeps it for
        the same reason this does: a number to score the model by, not a term.
        """
        with th.no_grad():
            return self.predict(input_data, highlight_mask.detach())

    def extra_logits(
        self, input_data: InputData, selection: th.Tensor
    ) -> Dict[str, th.Tensor]:
        """The two passes MRD trains on: the complement, and the full input."""
        return {
            "complement_class_logits": self.predict_complement(input_data, selection),
            "full_class_logits": self.predict_full(input_data),
        }
