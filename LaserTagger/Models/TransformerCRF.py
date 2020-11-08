from torch import nn
from Models.CRF import CRF
from Models.Transformer import Transformer
from Models.SequenceEncoder import SequenceEncoder
import torch

class Transformer_CRF(nn.Module):
    def __init__(self, vocab_size, ctx_vocab_size, nb_labels, emb_dim, hidden_dim, bos_idx, eos_idx, pad_idx, num_lstm_layers, dropout, device):
        super().__init__()
        self.transformer = Transformer(
            vocab_size, nb_labels=nb_labels, in_dim=emb_dim, dropout=dropout
        )
        self.crf = CRF(
            nb_labels,
            device,
            bos_idx,
            eos_idx,
            pad_tag_id=pad_idx,
            batch_first=True,
        )
        self.before_ctx_encoder = SequenceEncoder(ctx_vocab_size,  emb_dim, emb_dim, num_lstm_layers, dropout, 0)
        self.after_ctx_encoder = SequenceEncoder(ctx_vocab_size, emb_dim, emb_dim, num_lstm_layers, dropout, 0)
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

    def forward(self, x, before_ctx, after_ctx, mask=None):
        encoded_ctx = self.combine_ctx(x, before_ctx, after_ctx)
        emissions = self.transformer(x, encoded_ctx, mask)
        score, path = self.crf.decode(emissions, mask=mask)
        return score, path

    def loss(self, x,  before_ctx, after_ctx, y, mask=None):
        encoded_ctx = self.combine_ctx(x, before_ctx, after_ctx)
        emissions = self.transformer(x, encoded_ctx, mask)
        nll = self.crf(emissions, y, mask=mask)
        return nll
