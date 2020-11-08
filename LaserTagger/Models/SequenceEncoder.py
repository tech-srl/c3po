import torch.nn as nn
from Models.Embedding import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size,  in_dim, h_dim, num_layers, dropout, padding_idx):
        super(SequenceEncoder, self).__init__()
        self.emb = Embedding(vocab_size, in_dim, padding_idx)
        self.rnn = nn.LSTM(input_size=in_dim,
                                hidden_size=h_dim,
                                num_layers=num_layers,
                                dropout=dropout,
                                bidirectional=True)
        self.linear = nn.Linear(2 * h_dim, h_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.num_direction = 2
        self.h_dim = h_dim

    def forward(self, src):
        """
        :param src: of shape (src_length, batch)
        :return: out: PackedSequence of shape (batch, h_dim)
        """
        src, src_lengths = pad_packed_sequence(src)
        src_length, batch_size = src.size()
        # (src_length, batch, in_dim)
        src_in = self.emb(src)

        src_packed = pack_padded_sequence(src_in, src_lengths, enforce_sorted=False)
        # out: (path_length, batch, num_directions * h_dim)
        # (h_n, c_n): (num_layers * num_directions, batch, h_dim)
        out, (h_n, c_n) = self.rnn(src_packed)
        # (num_layers, num_directions, batch, h_dim)
        h_n = h_n.view(self.num_layers, self.num_direction, batch_size, self.h_dim)
        # (batch, h_dim)
        h_forward = h_n[-1, 0]
        h_backward = h_n[-1, 1]
        # (batch, 2* h_dim)
        out_cat = torch.cat((h_forward, h_backward), dim=1)
        # (batch, h_dim)
        out_linear = self.linear(out_cat)
        output = torch.tanh(out_linear)
        output = self.dropout(output)
        return output

