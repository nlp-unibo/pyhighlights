from typing import List

import torch as th
from torch.nn.functional import gumbel_softmax
from transformers import AutoModel

from pyhighlights.models.spp.base import SPP, SPPEmbedder, SPPEncoder, SPPSelector, SPPPredictor


class MGR(SPP):



class MGR(th.nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            classification_head,
            selection_head,
            num_generators,
            dropout_rate=0.0,
            embedding_matrix=None,
            freeze_embeddings=False,
            temperature=1.0
    ):
        super().__init__()

        self.temperature = temperature

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)
        self.dropout = th.nn.Dropout(p=dropout_rate)

        self.num_generators = num_generators
        self.generators = th.nn.ModuleList()
        for gen_idx in range(num_generators):
            self.generators.append(
                th.nn.Sequential(
                    th.nn.GRU(input_size=embedding_dim,
                              hidden_size=hidden_size,
                              batch_first=True,
                              bidirectional=True),
                    SelectItem(0),
                    th.nn.LayerNorm(hidden_size * 2),
                    self.dropout,
                    selection_head()
                ))

        self.classifier_encoder = th.nn.GRU(input_size=embedding_dim,
                                            hidden_size=hidden_size,
                                            batch_first=True,
                                            bidirectional=True)

        self.classification_head = classification_head()

    def generator(
            self,
            text,
            mask,
            generator_idx
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        # [bs, N, 2]
        highlight_logits = self.generators[generator_idx](tokens_emb)

        highlight_hat = gumbel_softmax(logits=highlight_logits,
                                       tau=self.temperature,
                                       hard=True)[:, :, 1]

        return highlight_hat, highlight_logits

    def classifier(
            self,
            text,
            mask,
            highlight_mask
    ):
        # [bs, N, d]
        hl_tokens_emb = self.embedding(text) * mask[:, :, None]
        hl_tokens_emb *= highlight_mask[:, :, None]

        # [bs, N, d'], [bs, d']
        hl_tokens_emb, _ = self.classifier_encoder(hl_tokens_emb)
        hl_tokens_emb = self.layer_norm(hl_tokens_emb)
        hl_tokens_emb = hl_tokens_emb * mask[:, :, None] + (1. - mask[:, :, None]) * (-1e6)
        hl_tokens_emb = th.transpose(hl_tokens_emb, 1, 2)

        # [bs, d']
        hl_emb, _ = th.max(hl_tokens_emb, dim=2)
        hl_emb = self.dropout(hl_emb)

        # [bs, #classes]
        logits = self.classification_head(hl_emb)

        return logits

    def forward(
            self,
            text,
            attention_mask,
            sample_ids
    ):
        hl_hat_list = []
        hl_logits_list = []
        logits_list = []

        for generator_idx in range(self.num_generators):
            # [bs, N]
            gen_hl_hat, gen_hl_logits = self.generator(text=text,
                                                       mask=attention_mask,
                                                       generator_idx=generator_idx)
            hl_hat_list.append(gen_hl_hat)
            hl_logits_list.append(gen_hl_logits)

            # [bs, #classes]
            logits = self.classifier(text=text,
                                     mask=attention_mask,
                                     highlight_mask=gen_hl_hat)
            logits_list.append(logits)

        return logits_list, hl_hat_list, hl_logits_list, attention_mask

    def forward_one_head(
            self,
            text,
            attention_mask,
            sample_ids
    ):
        # [bs, N]
        highlight_hat, highlight_logits = self.generator(text=text,
                                                         mask=attention_mask,
                                                         generator_idx=0)

        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=highlight_hat)

        return logits, highlight_hat, attention_mask