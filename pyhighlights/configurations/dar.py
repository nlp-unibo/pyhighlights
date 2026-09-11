"""DAR: a highlight scored by a module that only ever read the full input."""

from typing import List

from cinnamon.configuration import Param
from cinnamon.registry import RegistrationKey, register_class

from pyhighlights.components.models.spp.base import SPPBackbone
from pyhighlights.configurations.base import SPPModelConfig
from pyhighlights.configurations.keys import (
    ALIGNMENT_CLASSIFICATION_LOSS,
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    NAMESPACE,
    SPARSITY_LOSS,
    TRANSFORMER_BACKBONE,
)
from pyhighlights.utility.losses import Loss

DAR_COMPONENT = "pyhighlights.components.models.spp.dar.DAR"


@register_class(
    name="model", tags={"dar", "gru"}, namespace=NAMESPACE, component=DAR_COMPONENT
)
class GRUDARConfig(SPPModelConfig):
    name: str = Param("dar")
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
    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    #: Epochs the aligner spends on the full input before the first
    #: rationalization epoch. The reference implementation spends 100 and
    #: keeps the best of them against a validation split; this keeps the last,
    #: so the default is the smaller number a fixed budget can afford.
    pretrain_epochs: int = Param(20, ge=1)


@register_class(
    name="model",
    tags={"dar", "transformer"},
    namespace=NAMESPACE,
    component=DAR_COMPONENT,
)
class TransformerDARConfig(GRUDARConfig):
    selector_backbones: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    predictor_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
    aligner_backbone: RegistrationKey[SPPBackbone] = Param(TRANSFORMER_BACKBONE)
