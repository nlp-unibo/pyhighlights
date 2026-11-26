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
    sample_at_evaluation: bool = Param(
        True,
        description=(
            "Whether validation and test sample the hard Gumbel selection rather "
            "than take the argmax. DAR samples, as its original implementation "
            "does."
        ),
    )
    repair_empty_selections: bool = Param(
        False,
        description=(
            "Whether a selection that kept no word keeps its highest-scoring "
            "word instead. DAR does not repair, as its original implementation "
            "does not."
        ),
    )
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes for the predictor, "
            "separate from the aligner's because the predictor reads highlights "
            "and the aligner reads full text. One encoder trained on both is the "
            "drift DAR is about."
        ),
    )
    aligner_backbone: RegistrationKey[SPPBackbone] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the aligner's own backbone. The aligner's head "
            "is built from ``predictor``, since the two modules answer the same "
            "question about the same labels."
        ),
    )
    aligner_loss: RegistrationKey[Loss] = Param(
        ALIGNMENT_CLASSIFICATION_LOSS,
        description=(
            "Registration key of the loss that scores the aligner on the full "
            "input while it is pretrained, and on the highlight afterwards. The "
            "model appends it to ``losses``, so a registration cannot forget the "
            "term the method is."
        ),
    )
    pretrain_epochs: int = Param(
        20,
        ge=1,
        description=(
            "Number of epochs, at least one, the aligner trains on the full input "
            "before the first rationalization epoch. The aligner of the epoch "
            "with the lowest validation loss is kept, or that of the last epoch "
            "without validation."
        ),
    )


@register_class(key=TRANSFORMER_DAR, component=DAR_COMPONENT)
class TransformerDARConfig(GRUDARConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    aligner_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
