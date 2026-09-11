"""Field sets shared by model configurations.

Nothing here registers: every model would otherwise inherit a registration it
never asked for. The fields each model overrides -- the backbones, the losses,
the optimizer -- are named once here and pinned per model in ``fr``, ``mgr``,
``mcd``, ``mrd``, ``dr``, ``dar``, ``grat`` and ``genspp``.

Three bases rather than one, because the architectures disagree about what a
loss is: :class:`SPPModelConfig` scores a flat list,
:class:`PhasedSPPModelConfig` scores three lists tied to training phases, and
:class:`SPPShapeConfig` is what the two share.
"""

from typing import List

import torch as th
from cinnamon.configuration import Configuration, Param
from cinnamon.registry import RegistrationKey

from pyhighlights.components.models.spp.base import (
    SPPAggregator,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)
from pyhighlights.configurations.keys import (
    ADAM,
    CLASSIFICATION_LOSS,
    CONTIGUITY_LOSS,
    GRU_BACKBONE,
    MLP_PREDICTOR,
    MLP_SELECTOR,
    SPARSITY_LOSS,
)
from pyhighlights.utility.losses import Loss
from pyhighlights.utility.metrics import BoundMetric


class SPPShapeConfig(Configuration):
    """The parts every SPP model has, without saying what it optimises.

    Split from :class:`SPPModelConfig` because not every architecture has a
    flat ``losses`` list: MCD and MRD score their criteria per training phase
    and take three lists instead. Restating the other eleven fields to get rid
    of one is what cost PR #48 -- ``encoder_lr`` was added here and reached
    neither -- so the shape lives in one place and each model says only what
    differs.
    """

    name: str = Param("spp")
    selector_backbones: RegistrationKey[SPPBackbone] = Param(GRU_BACKBONE)
    selectors: RegistrationKey[SPPSelector] = Param(MLP_SELECTOR)
    predictor: RegistrationKey[SPPPredictor] = Param(MLP_PREDICTOR)
    predictor_backbone: RegistrationKey[SPPBackbone] | None = Param(None)
    aggregator: RegistrationKey[SPPAggregator] | None = Param(None)
    temperature: float = Param(1.0, gt=0.0)
    #: A selection is made over words: the unit the corpus annotates, that a
    #: sparsity target is a fraction of, and that an export shows -- the same
    #: unit whichever backbone read the text. ``"subtoken"`` is what the
    #: library did before, kept so the two can be compared. No ``variants``:
    #: that would expand every model into two keys for a switch almost nobody
    #: sweeps, and a study comparing them says so itself.
    select_over: str = Param("word")
    #: One rate for the encoders, another for everything above them. ``None``
    #: trains the whole model at the optimizer's own rate, which is what every
    #: published implementation of these architectures does -- they encode
    #: with a GRU over a frozen table, so nothing pretrained is fine-tuned.
    #: Set it when a pretrained encoder *is* being fine-tuned: one rate cannot
    #: serve both a transformer and a selector initialized from scratch.
    encoder_lr: float | None = Param(None, gt=0.0)
    optimizer: RegistrationKey[th.optim.Optimizer] = Param(ADAM)
    train_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    val_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)
    test_metrics: List[RegistrationKey[BoundMetric]] | None = Param(None)


class SPPModelConfig(SPPShapeConfig):
    """An SPP model scoring one flat list of criteria."""

    losses: List[RegistrationKey[Loss]] = Param(
        [CLASSIFICATION_LOSS, SPARSITY_LOSS, CONTIGUITY_LOSS]
    )


class PhasedSPPModelConfig(SPPShapeConfig):
    """An SPP model whose criteria belong to a training phase.

    MCD and MRD are the same shape and differ only in which criteria go in
    which list, so the lists are declared once. Both refuse
    ``supervise_highlights`` for the same reason: a supervision loss appended
    to a flat list would be dropped before the first batch, and the phase it
    belongs to has to be named.
    """

    rationale_losses: List[RegistrationKey[Loss]] = Param(
        [SPARSITY_LOSS, CONTIGUITY_LOSS]
    )
    predictor_losses: List[RegistrationKey[Loss]] = Param([])
    generator_losses: List[RegistrationKey[Loss]] = Param([])
