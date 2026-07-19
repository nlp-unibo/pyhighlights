import torch as th

class AttentionPooling(th.nn.Module):
    def __init__(
            self,
            in_dim: int,
            hidden_size: int):
        """
        AttentionPoolingBlock
        """
        super().__init__()
        self.attention = th.nn.Sequential(
            th.nn.Linear(in_dim, hidden_size),
            th.nn.LayerNorm(hidden_size),
            th.nn.GELU(),
            th.nn.Linear(hidden_size, 1)
        )
        self._weights = None

    @property
    def attn_weights(
            self
    ):
        return self._weights

    def forward(
            self,
            hidden_states: th.Tensor,
            mask: th.Tensor | None = None
    ):
        """
        :param hidden_states: [batch,seq_len,dim]
        :param mask: bool tensor [batch,seq_len]
        :return: converge_representations [batch,dim]
        """
        weight = self.attention(hidden_states).squeeze()
        if mask is not None:
            mask = mask.bool()
            weight.masked_fill_(~mask, th.finfo(th.float).min)
        weight = th.nn.functional.softmax(weight, dim=-1)
        self._weights = weight
        weight = weight.unsqueeze(dim=1)
        converge_representations = th.bmm(weight, hidden_states).squeeze()
        return converge_representations


class FactorAnnealer:
    def __init__(
            self,
            factor: float,
            decay_callback
    ):
        self.factor = factor
        self.decay_callback = decay_callback
        self.current_step = 0
        self._current_factor = factor

    def step(self):
        decay_factor = self.decay_callback(self.current_step)
        self.current_step += 1
        self._current_factor = self.factor * decay_factor
        return self._current_factor

    @property
    def current_factor(self):
        return self._current_factor