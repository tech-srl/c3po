from torch import nn
from Models.CRF import CRF
from Models.Transformer import Transformer
from Models.TransformerCtx import TransformerCtx
from Models.SequenceEncoder import SequenceEncoder
from Models.Attention import Attention
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Transformer_CRF(nn.Module):
    def __init__(self, vocab_size, ctx_vocab_size, nb_labels, emb_dim, hidden_dim, bos_idx, eos_idx, pad_idx, num_lstm_layers, dropout, device):
        super().__init__()
        self.transformer = Transformer(
            vocab_size, in_dim=emb_dim, nb_labels=nb_labels, dropout=dropout
        )
        self.crf = CRF(
            nb_labels,
            device,
            bos_idx,
            eos_idx,
            pad_tag_id=pad_idx,
            batch_first=True,
        )
        self.ctx_encoder = TransformerCtx(ctx_vocab_size, device=device, in_dim=emb_dim)
        self.ctx_combiner = Attention(emb_dim)
        self.query = nn.Parameter(torch.Tensor(1, emb_dim))
        torch.nn.init.xavier_uniform_(self.query.data)
        self.emb_dim = emb_dim
        self.ctx_linear = nn.Linear(2 * emb_dim, emb_dim)

    def combine_ctx(self, x, before_ctx, after_ctx):
        # (batch, h_dim)
        before_ctx_encoded = self.before_ctx_encoder(before_ctx)
        after_ctx_encoded = self.after_ctx_encoder(after_ctx)

        # (batch, 2 * h_dim)
        ctx_cat = torch.cat((before_ctx_encoded, after_ctx_encoded), dim=1)
        # (batch, h_dim)
        encoded_ctx = torch.tanh(self.ctx_linear(ctx_cat))

        seq_len = x.shape[1]

        # (batch, seq_len, h_dim)
        encoded_ctx_repeated = encoded_ctx.unsqueeze(dim=0).repeat(seq_len, 1, 1)
        return encoded_ctx_repeated

    def forward_ctx(self, x, before_ctx, after_ctx):
        batch_size = x.shape[0]
        # (batch_size, 1, emb_dim)
        query = self.query.expand(batch_size, self.emb_dim).unsqueeze(dim=1)
        packed_query = pack_padded_sequence(query, batch_size * [1], batch_first=True, enforce_sorted=False)
        # Packed sequence (before_ctx_length, batch_size, emb_dim)
        encoded_before_ctx = self.ctx_encoder(before_ctx)
        # (batch_size, 1, emb_dim)
        encoded_before_ctx, _ = self.ctx_combiner(packed_query, encoded_before_ctx)
        # Packed sequence (after_ctx_length, batch_size, emb_dim)
        encoded_after_ctx = self.ctx_encoder(after_ctx)
        # (batch_size, 1 ,emb_dim)
        encoded_after_ctx, _ = self.ctx_combiner(packed_query, encoded_after_ctx)
        # (batch_size ,emb_dim)
        combined_ctx = self.ctx_linear(torch.cat([encoded_before_ctx, encoded_after_ctx], dim=2).squeeze())
        # (1, batch_size ,emb_dim)
        combined_ctx = combined_ctx.unsqueeze(dim=0)
        seq_len = x.shape[1]
        # (seq_len, batch_size, emb_dim)
        combined_ctx = combined_ctx.repeat(seq_len, 1, 1)
        return combined_ctx

    def forward(self, x, before_ctx, after_ctx, mask=None):
        # (seq_len, batch_size, emb_dim)
        combined_ctx = self.forward_ctx(x, before_ctx, after_ctx)
        # (batch_size, src_length, num_labels)
        emissions = self.transformer(x, combined_ctx, mask)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x,  before_ctx, after_ctx, y, mask=None):
        # (seq_len, batch_size, emb_dim)
        combined_ctx = self.forward_ctx(x, before_ctx, after_ctx)
        # (batch_size, src_length, num_labels)
        emissions = self.transformer(x, combined_ctx, mask)
        nll = self.crf(emissions, y, mask=mask)
        return nll
