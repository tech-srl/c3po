import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class SequenceEncoder(nn.Module):
    def __init__(self,  in_dim, h_dim, num_layers, dropout):
        super(SequenceEncoder, self).__init__()
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
        # out: (path_length, batch, num_directions * h_dim)
        # (h_n, c_n): (num_layers * num_directions, batch, h_dim)
        packed_lstm_out, (h_n, c_n) = self.rnn(src)

        padded_lstm_out, lstm_out_lengths = pad_packed_sequence(packed_lstm_out, batch_first=True)
        # (batch_size, seq_len, h_dim)
        out_linear = self.linear(padded_lstm_out)
        out = torch.tanh(out_linear)
        out = self.dropout(out)
        # (batch_size, h_dim)
        #avg_out = out.sum(dim=1).div(lstm_out_lengths.float().unsqueeze(dim=1))
        packed_out = pack_padded_sequence(out, lstm_out_lengths, batch_first=True, enforce_sorted=False)
        return packed_out#, avg_out

