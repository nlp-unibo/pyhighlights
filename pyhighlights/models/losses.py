import torch as th


class SparsityConstraint(th.nn.Module):

    def __init__(
            self,
            sparsity_level
    ):
        super().__init__()

        self.sparsity_level = sparsity_level

    def forward(
            self,
            highlight_hat,
            attention_mask,
    ):
        # highlight_hat:    [bs, N]
        # attention_mask:   [bs, N]

        # [bs]
        sparsity = th.sum(highlight_hat) / th.sum(attention_mask)
        return th.abs(sparsity - self.sparsity_level)


class ContinuityConstraint(th.nn.Module):

    def forward(
            self,
            highlight_mask
    ):
        # highlight_mask:    [bs, N]

        return th.mean(th.abs(highlight_mask[:, 1:] - highlight_mask[:, :-1]))


class JS_Div(th.nn.Module):
    def __init__(
            self
    ):
        super(JS_Div, self).__init__()
        self.kl_div = th.nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, p, q):
        p_s = th.nn.functional.softmax(p, dim=-1)
        q_s = th.nn.functional.softmax(q, dim=-1)
        p_s, q_s = p_s.view(-1, p_s.size(-1)), q_s.view(-1, q_s.size(-1))
        m = (0.5 * (p_s + q_s)).log()
        return 0.5 * (self.kl_div(m, p_s.log()) + self.kl_div(m, q_s.log()))


class KL_Div(th.nn.Module):

    def forward(
            self,
            p,
            q
    ):
        return th.nn.functional.kl_div(th.nn.functional.softmax(p, dim=-1).log(),
                                       th.nn.functional.softmax(q, dim=-1),
                                       reduction='batchmean')