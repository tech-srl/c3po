import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=22):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class Transformer(nn.Module):
    def __init__(self, device, in_dim=512, num_head=8, num_layers=4, dropout=0.25):
        super(Transformer, self).__init__()
        h_dim = 4 * in_dim
        encoder_layers = TransformerEncoderLayer(in_dim, num_head, h_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        self.positional_encoder = PositionalEncoding(in_dim)
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.device = device

    def forward(self, src):
        """
        :param src: PackedSequence of shape (src_length, batch_size, in_dim)
        :return: 'mixed' src of shape (src_length, batch_size, h_dim)
        """
        # (src_length, batch_size, in_dim)
        src, src_lengths = pad_packed_sequence(src)
        src = self.positional_encoder(src)
        max_len = src.shape[0]
        # src_key_padding_mask = torch.arange(max_len).expand(len(src_lengths), max_len) >= src_lengths.unsqueeze(1)
        src_key_padding_mask = (torch.arange(max_len)[None, :] >= src_lengths[:, None]).to(self.device)
        # (src_length, batch_size, h_dim)
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        output = pack_padded_sequence(output, src_lengths, enforce_sorted=False)
        return output
