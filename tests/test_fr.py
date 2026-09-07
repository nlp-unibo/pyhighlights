import torch as th
from cinnamon.configuration import Configuration
from cinnamon.registry import RegistrationKey, Registry

from pyhighlights.components.models import InputData
from pyhighlights.components.models.spp import (
    FR,
    SPPBackbone,
    SPPPredictor,
    SPPSelector,
)

NAMESPACE = "tests"


class TinyBackbone(SPPBackbone):
    def __init__(self):
        super().__init__()
        self.embedding = th.nn.Embedding(8, 4)

    @property
    def output_size(self) -> int:
        return 4

    def encode(
        self,
        features: th.Tensor,
        mask: th.Tensor,
        selection_mask: th.Tensor | None = None,
    ) -> th.Tensor:
        selected = mask if selection_mask is None else mask * selection_mask
        return self.embedding(features) * selected.unsqueeze(-1)

    def pool(self, states: th.Tensor, mask: th.Tensor) -> th.Tensor:
        mask = mask.unsqueeze(-1)
        return (states * mask).sum(1) / mask.sum(1).clamp_min(1)


class TinySelector(SPPSelector):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = th.nn.Linear(input_size, 2)

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.linear(states)


class TinyPredictor(SPPPredictor):
    def __init__(self, input_size: int):
        super().__init__()
        self.linear = th.nn.Linear(input_size, 3)

    def forward(self, states: th.Tensor) -> th.Tensor:
        return self.linear(states)


def register(name: str, component: str) -> RegistrationKey:
    return Registry.register_configuration(
        config=Configuration.default(),
        name=name,
        namespace=NAMESPACE,
        component=component,
    )


def test_fr_backend_contract_and_deterministic_evaluation():
    Registry.initialize()
    backbone = register("backbone", f"{__name__}.TinyBackbone")
    selector = register("selector", f"{__name__}.TinySelector")
    predictor = register("predictor", f"{__name__}.TinyPredictor")
    Registry.dag_resolution()

    model = FR(
        name="fr",
        losses=[],
        optimizer=RegistrationKey(name="optimizer", namespace=NAMESPACE),
        selector_backbones=backbone,
        selectors=selector,
        predictor=predictor,
    )
    with th.no_grad():
        model.selectors[0].linear.weight.zero_()
        model.selectors[0].linear.bias.copy_(th.tensor([2.0, -2.0]))
        model.predictor.linear.weight.fill_(1.0)

    batch = InputData(
        features=th.tensor([[1, 2, 3], [4, 5, 0]]),
        mask=th.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]),
        sample_ids=th.arange(2),
        y_true=th.tensor([0, 1]),
        highlight_true=th.full((2, 3), -1),
    )

    model.eval()
    first = model(batch)
    second = model(batch)

    assert first.class_logits.shape == (2, 1, 3)
    assert first.highlight_logits.shape == (2, 1, 3, 2)
    assert first.highlight_mask.shape == (2, 1, 3)
    assert th.equal(first.highlight_mask, second.highlight_mask)
    assert first.highlight_mask[1, 0, 2] == 0
    assert th.allclose(first.highlight_mask.sum(dim=-1), th.ones((2, 1)))
    assert model.selector_backbone is model.predictor_backbone

    first.class_logits.sum().backward()
    assert model.selector_backbone.embedding.weight.grad is not None
    assert model.selectors[0].linear.bias.grad.abs().sum() > 0
