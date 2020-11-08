import torch.nn as nn
from Models.Embedding import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class SequenceTagger(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, in_dim, h_dim, num_layers, dropout, padding_idx=0):
        super(SequenceTagger, self).__init__()
        self.num_direction = 2
        self.emb = Embedding(src_vocab_size, in_dim, padding_idx)
        self.rnn = nn.LSTM(input_size=in_dim,
                           hidden_size=h_dim,
                           num_layers=num_layers,
                           dropout=dropout,
                           bidirectional=True if self.num_direction == 2 else False)
        self.linear = nn.Linear(self.num_direction * h_dim, tgt_vocab_size)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.h_dim = h_dim

    def forward(self, src):
        """
        :param src: of shape (src_length, batch)
        :return: out: PackedSequence of shape (batch, h_dim)
        """
        src, src_lengths = pad_packed_sequence(src)
        # (src_length, batch, in_dim)
        src_in = self.emb(src)

        src_packed = pack_padded_sequence(src_in, src_lengths, enforce_sorted=False)
        # lstm_out: PackedSequence of shape (path_length, batch, num_directions * h_dim)
        # (h_n, c_n): (num_layers * num_directions, batch, h_dim)
        lstm_out, (h_n, c_n) = self.rnn(src_packed)
        # (batch_size, inputs_length , 2 * h_dim)
        padded_lstm_out, lstm_out_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        # (batch, inputs_length, tgt_vocab_size)
        out_padded = self.linear(padded_lstm_out)

        out_packed = pack_padded_sequence(out_padded, lstm_out_lengths, enforce_sorted=False, batch_first=True)
        return out_packed

