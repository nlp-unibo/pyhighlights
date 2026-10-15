import torch as th

from pyhighlights.components.models.spp.implementations import GRUBackbone


def test_a_gru_backbone_without_layer_norm_returns_the_recurrent_states():
    th.manual_seed(0)
    plain = GRUBackbone(vocab_size=6, embedding_dim=3, hidden_size=4, layer_norm=False)
    normalised = GRUBackbone(vocab_size=6, embedding_dim=3, hidden_size=4)
    features = th.tensor([[2, 3, 4], [5, 2, 0]])
    mask = th.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]])

    states = plain.encode(features, mask)
    expected, _ = plain.encoder(plain.embedding(features[:1]))

    assert not any("layer_norm" in name for name, _ in plain.named_parameters())
    assert any("layer_norm" in name for name, _ in normalised.named_parameters())
    assert th.allclose(states[:1], expected)
    assert th.equal(states[1, 2], th.zeros(8))
