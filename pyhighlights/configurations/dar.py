"""DAR: a highlight scored by a module that only ever read the full input."""

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    ALIGNMENT_CLASSIFICATION_LOSS,
    GRU_BACKBONE,
    GRU_DAR,
    TRANSFORMER_BACKBONE,
    TRANSFORMER_DAR,
)
from pyhighlights.utility.losses import Loss

DAR_COMPONENT = "pyhighlights.components.models.spp.dar.DAR"


@register_class(key=GRU_DAR, component=DAR_COMPONENT)
class GRUDARConfig(SPPModelConfig):
    name: str = Param("dar")
    #: As the original implementation: a sampled selection at evaluation and
    #: no empty-selection repair.
    sample_at_evaluation: bool = Param(True)
    repair_empty_selections: bool = Param(False)
    #: Separate encoders: the predictor reads highlights and the aligner reads
    #: full text, and one encoder trained on both is the drift DAR is about.
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    #: The aligner's own encoder. Its head is built from ``predictor``: the
    #: two modules answer the same question about the same labels.
    aligner_backbone: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    #: Scores the aligner on the full input while it is pretrained, and on the
    #: highlight afterwards. Appended to ``losses`` by the model, so a
    #: registration cannot forget the term the method is.
    aligner_loss: RegistrationKey[Loss] = Param(ALIGNMENT_CLASSIFICATION_LOSS)
    #: Epochs the aligner spends on the full input before the first
    #: rationalization epoch. The aligner of the epoch with the lowest
    #: validation loss is kept, or that of the last epoch without validation.
    pretrain_epochs: int = Param(20, ge=1)


@register_class(key=TRANSFORMER_DAR, component=DAR_COMPONENT)
class TransformerDARConfig(GRUDARConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    aligner_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
