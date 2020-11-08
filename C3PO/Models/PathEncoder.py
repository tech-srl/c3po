import torch.nn as nn
from Models.Embedding import Embedding
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch


class PathEncoder(nn.Module):
    def __init__(self, path_vocab_size, src_tgt_vocab_size, position_vocab_size, in_dim, h_dim, num_layers, dropout, padding_idx):
        super(PathEncoder, self).__init__()
        self.src_tgt_emb = Embedding(src_tgt_vocab_size, in_dim, padding_idx)
        self.path_emb = Embedding(path_vocab_size, in_dim, padding_idx)
        self.pos_emb = Embedding(position_vocab_size, in_dim, padding_idx)
        self.path_rnn = nn.LSTM(input_size=in_dim,
                                hidden_size=h_dim,
                                num_layers=num_layers,
                                dropout=dropout,
                                bidirectional=False)
        self.linear = nn.Linear(h_dim + 2 * in_dim, h_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.num_layers = num_layers
        self.h_dim = h_dim

    def forward(self, src, src_pos, tgt, tgt_pos, path, path_pos):
        """
        :param src: PackedSequence of shape (src_length, batch)
        :param src_pos: PackedSequence of shape (src_length, batch)
        :param tgt: PackedSequence of shape (tgt_length, batch)
        :param tgt_pos: PackedSequence of shape (tgt_length, batch)
        :param path: PackedSequence of shape (path_length, batch)
        :param path_pos: PackedSequence of shape (path_length, batch)
        :return: out: of shape (batch, h_dim)
        """
        src, src_lengths = pad_packed_sequence(src)
        src_pos, src_pos_lengths = pad_packed_sequence(src_pos)
        tgt, tgt_lengths = pad_packed_sequence(tgt)
        tgt_pos, tgt_pos_lengths = pad_packed_sequence(tgt_pos)
        path, path_lengths = pad_packed_sequence(path)
        path_pos, path_pos_lengths = pad_packed_sequence(path_pos)

        src_length, _ = src.size()
        tgt_length, _ = tgt.size()
        path_length, batch_size = path.size()

        # (src_length + tgt_length, batch)
        src_tgt = torch.cat((src, tgt), dim=0)
        # (src_length + tgt_length, batch, in_dim)
        src_tgt_in = self.src_tgt_emb(src_tgt)
        # (src_length, batch, in_dim)
        # (tgt_length, batch, in_dim)
        src_in, tgt_in = torch.split(src_tgt_in, [src_length, tgt_length], dim=0)
        # (src_pos_length + tgt_pos_length + path_pos_length, batch)
        positions = torch.cat((src_pos, tgt_pos, path_pos))
        # (src_pos_length + tgt_pos_length + path_pos_length, batch, in_dim)
        pos_in = self.pos_emb(positions)
        # (src_length, batch, in_dim)
        # (tgt_length, batch, in_dim)
        # (path_length, batch, in_dim)
        src_pos_in, tgt_pos_in, path_pos_in = torch.split(pos_in, [src_length, tgt_length, path_length], dim=0)

        # (path_length, batch, in_dim)
        path_in = self.path_emb(path)

        # add in_emb and pos_emb
        # (path_length, batch, in_dim)
        src_sum_in = src_in + src_pos_in
        tgt_sum_in = tgt_in + tgt_pos_in
        path_sum_in = path_in + path_pos_in

        # (batch, in_dim)
        src_out = torch.sum(src_sum_in, dim=0)
        tgt_out = torch.sum(tgt_sum_in, dim=0)

        path_packed = pack_padded_sequence(path_sum_in, path_lengths, enforce_sorted=False)

        # path_out: (path_length, batch, h_dim)
        # (path_h_n, path_c_n): (num_layers, batch, h_dim)
        path_out, (path_h_n, path_c_n) = self.path_rnn(path_packed)
        # (batch, h_dim)
        h_forward = path_h_n[-1]

        # (batch, h_dim + 2 * in_dim)
        out_cat = torch.cat((h_forward, src_out, tgt_out), dim=1)
        # (batch, h_dim)
        output = torch.tanh(self.linear(out_cat))
        # (batch, h_dim)
        output = self.dropout(output)
        return output

