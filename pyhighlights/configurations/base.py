"""Field sets shared by model configurations.

Nothing here registers: every model would otherwise inherit a registration it
never asked for. The fields each model overrides, such as the backbones, the
losses and the optimizer, are named once here and pinned per model in ``fr``, ``mgr``,
``mcd``, ``mrd``, ``dr``, ``dar``, ``grat`` and ``genspp``.

Three bases rather than one, because the architectures disagree about what a
loss is: :class:`SPPModelConfig` scores a flat list,
:class:`PhasedSPPModelConfig` scores three lists tied to training phases, and
:class:`SPPShapeConfig` is what the two share.

Each model registers a GRU configuration and a Transformer subclass of it,
rather than declaring the Transformer backbone as a variant. A variant varies
one field, and for most models the switch changes two to four fields
together: every backbone, and for G-RAT its guider and ``share_layer_norm`` as
well. Variants on each field would build every mixed combination, and the
``one_token_space`` condition would drop all of them, 12 keys for G-RAT. FR
and the grounded model switch one field, but their variant key would keep the
``gru`` tag of the base key, so they follow the same pattern.
"""

from typing import TYPE_CHECKING, List, Literal

from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey

from pyhighlights.configurations.keys import (
    ADAM,
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    SPARSITY_LOSS,
)
from pyhighlights.utility.spaces import token_spaces

if TYPE_CHECKING:
    import torch

    import pyhighlights.components.models.spp.base
    import pyhighlights.utility.losses
    import pyhighlights.utility.metrics


class SPPShapeConfig(Configuration):
    """The parts every SPP model has, without saying what it optimises.

    Split from :class:`SPPModelConfig` because not every architecture has a
    flat ``losses`` list: MCD and MRD score their criteria per training phase
    and take three lists instead. The shape lives in one place, so a field
    added here reaches every model, and each model says only what differs.
    """

    name: str = Param("spp", description="Name of the model.")
    selector_backbones: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPBackbone"
    ] = Param(
        GRU_BACKBONE,
        description=(
            "Registration key of the backbone that encodes the input for the "
            "selector, or one key per selector for a model with several."
        ),
    )
    selectors: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPSelector"
    ] = Param(
        MLP_SELECTOR,
        description=(
            "Registration key of the selector that scores each position as kept "
            "or dropped, or one key per selector for a model with several."
        ),
    )
    predictor: RegistrationKey[
        "pyhighlights.components.models.spp.base.SPPPredictor"
    ] = Param(
        MLP_PREDICTOR,
        description=(
            "Registration key of the predictor that maps the pooled state of the "
            "highlight to class logits."
        ),
    )
    predictor_backbone: (
        RegistrationKey["pyhighlights.components.models.spp.base.SPPBackbone"] | None
    ) = Param(
        None,
        description=(
            "Registration key of the backbone that encodes the highlight for the "
            "predictor, or None to share the first selector backbone."
        ),
    )
    temperature: float = Param(
        1.0,
        gt=0.0,
        description=(
            "Temperature of the Gumbel-softmax that samples the selection, "
            "greater than zero."
        ),
    )
    compact: bool = Param(
        False,
        description=(
            "Whether the predictor reads the kept positions gathered into a "
            "shorter sequence instead of the dropped ones zeroed in place, which "
            "stops a selector from signalling a label through the shape of the "
            "mask but not through its count. It is off by default, because it "
            "changes what the predictor is trained on rather than fixing what it "
            "reads."
        ),
    )
    sample_at_evaluation: bool = Param(
        False,
        description=(
            "Whether validation and test sample the hard Gumbel selection as "
            "training does, rather than take the argmax. The original FR, MCD, "
            "MGR and DAR sample."
        ),
    )
    repair_empty_selections: bool = Param(
        True,
        description=(
            "Whether a selection that kept no word keeps its highest-scoring "
            "word instead. The original FR, MCD, MGR, G-RAT and DAR have no "
            "such repair."
        ),
    )
    select_over: Literal["word", "subtoken"] = Param(
        "word",
        description=(
            "Unit the selection is made over: ``word``, the unit the corpus "
            "annotates, that a sparsity target is a fraction of, and that an "
            "export shows, whichever backbone read the text. ``subtoken`` "
            "selects over the backbone's own tokens instead."
        ),
    )
    encoder_lr: float | None = Param(
        None,
        gt=0.0,
        description=(
            "Learning rate of the backbones, with everything above them at the "
            "optimizer's own rate, or None to train the whole model at that "
            "rate. Set it when a pretrained encoder is fine-tuned, since one "
            "rate cannot serve both a transformer and a selector initialized "
            "from scratch."
        ),
    )
    optimizer: RegistrationKey["torch.optim.Optimizer"] = Param(
        ADAM, description="Registration key of the optimizer that trains the model."
    )
    train_metrics: (
        List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] | None
    ) = Param(
        None,
        description=(
            "Registration keys of the metrics scored on the training split, or "
            "None for none."
        ),
    )
    val_metrics: (
        List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] | None
    ) = Param(
        None,
        description=(
            "Registration keys of the metrics scored on the validation split, or "
            "None for none."
        ),
    )
    test_metrics: (
        List[RegistrationKey["pyhighlights.utility.metrics.BoundMetric"]] | None
    ) = Param(
        None,
        description=(
            "Registration keys of the metrics scored on the test split, or None "
            "for none."
        ),
    )

    @classmethod
    def default(cls):
        config = super().default()
        # A condition rather than one backbone field, so a model whose
        # selector, predictor and guider are separate encoders keeps them
        # separate. Only their token space has to agree, and a grid that
        # varies the backbones independently drops the cells where it does not.
        config.add_condition(
            name="one_token_space",
            description=(
                "Every encoder of a model reads the same token ids: one "
                "pretrained model card, or the vocabulary the task fits."
            ),
            condition=lambda model: len(token_spaces(model)) <= 1,
        )
        return config


class SPPModelConfig(SPPShapeConfig):
    """An SPP model scoring one flat list of criteria."""

    losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS],
        description=(
            "Registration keys of the losses whose weighted sum the model minimizes."
        ),
    )


class PhasedSPPModelConfig(SPPShapeConfig):
    """An SPP model whose criteria belong to a training phase.

    MCD and MRD are the same shape and differ only in which criteria go in
    which list, so the lists are declared once. Both refuse
    ``supervise_highlights`` for the same reason: a supervision loss appended
    to a flat list would be dropped before the first batch, and the phase it
    belongs to has to be named.
    """

    shared_losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [SPARSITY_LOSS, CONTIGUITY_LOSS],
        description="Registration keys of the losses scored in both training phases.",
    )
    predictor_losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [],
        description=(
            "Registration keys of the losses scored in the predictor phase only."
        ),
    )
    generator_losses: List[RegistrationKey["pyhighlights.utility.losses.Loss"]] = Param(
        [],
        description=(
            "Registration keys of the losses scored in the generator phase only."
        ),
    )
    monitored_losses: List[str] | None = Param(
        None,
        description=(
            "Names of the losses whose weighted sum evaluation logs as ``loss``, "
            "or None to sum all of them."
        ),
    )
