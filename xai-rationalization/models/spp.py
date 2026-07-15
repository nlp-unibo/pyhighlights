import torch as th
from modeling.layers import SelectItem, AttentionPooling
from torch.nn.functional import gumbel_softmax


class FR(th.nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            classification_head,
            selection_head,
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

        self.encoder = th.nn.GRU(input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 batch_first=True,
                                 bidirectional=True)

        self.selection_head = selection_head()

        self.classification_head = classification_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)

        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

    def generator(
            self,
            text,
            mask
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        # [bs, N, d'], [bs, d']
        tokens_emb, _ = self.encoder(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb)
        tokens_emb = self.dropout(tokens_emb)

        # [bs, N, 2]
        highlight_logits = self.selection_head(tokens_emb)
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
        hl_tokens_emb, _ = self.encoder(hl_tokens_emb)
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
        # [bs, N]
        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=highlight_hat)

        return logits, highlight_hat, highlight_logits, attention_mask


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


class MCD(th.nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            classification_head,
            selection_head,
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

        self.gen_encoder = th.nn.GRU(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True)

        self.cls_encoder = th.nn.GRU(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     batch_first=True,
                                     num_layers=1,
                                     bidirectional=True)

        self.classification_head = classification_head()

        self.gen_classification_head = selection_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

        self.gen = th.nn.Sequential(
            self.gen_encoder,
            SelectItem(0),
            self.layer_norm,
            self.dropout,
            self.gen_classification_head
        )

    def generator(
            self,
            text,
            mask
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        # [bs, N, 2]
        highlight_logits = self.gen(tokens_emb)
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
        hl_tokens_emb, _ = self.cls_encoder(hl_tokens_emb)
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
        # [bs, N, 2], [bs, N]
        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=highlight_hat)

        return logits, highlight_hat, attention_mask

    def generator_forward(
            self,
            text,
            attention_mask,
            sample_ids
    ):
        # [bs, N]
        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        return highlight_hat, highlight_logits

    def classifier_forward(
            self,
            text,
            attention_mask,
            highlight_mask,
            sample_ids
    ):
        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=highlight_mask)
        return logits

    def no_selection_forward(
            self,
            text,
            attention_mask,
            sample_ids
    ):
        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=attention_mask)
        return logits


class GRAT(th.nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            classification_head,
            selection_head,
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

        self.gen_encoder = th.nn.GRU(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True)

        self.cls_encoder = th.nn.GRU(input_size=embedding_dim,
                                     hidden_size=hidden_size,
                                     batch_first=True,
                                     num_layers=1,
                                     bidirectional=True)

        self.classification_head = classification_head()

        self.gen_classification_head = selection_head()

        self.dropout = th.nn.Dropout(p=dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

        self.gen = th.nn.Sequential(
            self.gen_encoder,
            SelectItem(0),
            self.layer_norm,
            self.dropout,
            self.gen_classification_head
        )

    def generator(
            self,
            text,
            mask
    ):
        # [bs, N, d]
        tokens_emb = self.embedding(text)
        tokens_emb *= mask[:, :, None]

        # [bs, N, 2]
        highlight_logits = self.gen(tokens_emb)
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
        hl_tokens_emb, _ = self.cls_encoder(hl_tokens_emb)
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
        # [bs, N], [bs, N, 2]
        highlight_hat, highlight_logits = self.generator(text=text, mask=attention_mask)

        # [bs, #classes]
        logits = self.classifier(text=text,
                                 mask=attention_mask,
                                 highlight_mask=highlight_hat)

        return logits, highlight_hat, highlight_logits, attention_mask


class GRATGuider(th.nn.Module):

    def __init__(
            self,
            vocab_size,
            embedding_dim,
            hidden_size,
            classification_head,
            noise_sigma=1.0,
            dropout_rate=0.0,
            embedding_matrix=None,
            freeze_embeddings=False,
            temperature=1.0
    ):
        super().__init__()

        self.temperature = temperature
        self.noise_sigma = noise_sigma

        self.embedding = th.nn.Embedding(num_embeddings=vocab_size,
                                         embedding_dim=embedding_dim)
        if embedding_matrix is not None:
            self.embedding.weight.data = embedding_matrix

        if freeze_embeddings:
            self.embedding.weight.requires_grad = False

        self.activation = th.nn.GELU()
        self.convert_layer = th.nn.Linear(embedding_dim, hidden_size * 2, bias=True)
        self.encoder = th.nn.GRU(input_size=embedding_dim,
                                 hidden_size=hidden_size,
                                 num_layers=1,
                                 batch_first=True,
                                 bidirectional=True)
        self.attention_fc = th.nn.Sequential(
            th.nn.Linear(hidden_size * 2, hidden_size * 2),
            th.nn.GELU(),
            th.nn.Linear(hidden_size * 2, 1)
        )

        self.projection_fc = th.nn.Linear(hidden_size * 2, hidden_size * 2)
        self.pooling = AttentionPooling(hidden_size * 2, hidden_size * 2)
        self.out_head = classification_head()
        self.dropout = th.nn.Dropout(dropout_rate)
        self.layer_norm = th.nn.LayerNorm(hidden_size * 2)

    def forward(
            self,
            text,
            attention_mask,
            sample_ids
    ):
        bool_mask = attention_mask.to(th.bool)

        tokens_emb = self.embedding(text) * attention_mask[:, :, None]
        input_states = self.convert_layer(tokens_emb)

        tokens_emb, _ = self.encoder(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb + input_states)
        tokens_emb = self.dropout(tokens_emb)

        attention_weights = self.attention_fc(tokens_emb)
        attention_weights = attention_weights.masked_fill_(~bool_mask[:, :, None], th.finfo(th.float).min)

        if self.training:
            attention_noises = th.normal(0, self.noise_sigma, attention_weights.size(), device=attention_weights.device,
                                         dtype=attention_weights.dtype)
            attention_noises = th.abs(attention_noises).masked_fill_(~bool_mask[:, :, None], th.finfo(th.float).min)
            attention_weights += attention_noises

        attention_weights = th.nn.functional.softmax(attention_weights, dim=1)
        tokens_emb = self.projection_fc(attention_weights * tokens_emb)
        tokens_emb = self.activation(tokens_emb)
        tokens_emb = self.layer_norm(tokens_emb)

        final_states = self.pooling(tokens_emb, attention_mask)
        final_states = self.dropout(final_states)
        final_logits = self.out_head(final_states)

        return attention_weights, final_logits