import torch.nn as nn
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
import Constants


class OperationMix(nn.Module):
    def __init__(self, h_dim, dropout):
        super(OperationMix, self).__init__()
        self.mov_linear = nn.Linear(h_dim, h_dim, bias=False)
        self.upd_linear = nn.Linear(h_dim, h_dim, bias=False)
        self.ins_linear = nn.Linear(h_dim, h_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)
        self.EOS = nn.Parameter(torch.Tensor(1, h_dim))
        self.PAD = nn.Parameter(torch.Tensor(1, h_dim))
        torch.nn.init.xavier_uniform_(self.EOS.data)
        torch.nn.init.xavier_uniform_(self.PAD.data)
        self.h_dim = h_dim

    def forward(self, encoder_outputse):
        """
        :param encoder_outputs: PackedSequence of shape (encoded_seq_len, batch_size, h_dim)
        :return: output: PackedSequence of shape (2 + 3 * encoded_seq_len, batch_size, h_dim)
        """
        # (encoded_seq_len, batch_size, h_dim)
        src, src_lengths = pad_packed_sequence(encoder_outputse)
        encoded_seq_len, batch_size, _ = src.size()

        # (1, batch_size, h_dim)
        eos = self.EOS.expand(batch_size, self.h_dim).unsqueeze(dim=0)
        pad = self.PAD.expand(batch_size, self.h_dim).unsqueeze(dim=0)
        # eos = self.EOS.repeat(batch_size, 1).unsqueeze(dim=0)
        # pad = self.PAD.repeat(batch_size, 1).unsqueeze(dim=0)

        # (NUM_OF_OPS * encoded_seq_len, batch_size, h_dim)
        combined = src.new_zeros(size=(Constants.NUM_OF_OPS * encoded_seq_len, batch_size, self.h_dim))

        # (encoded_seq_len, batch_size, h_dim)
        combined[:encoded_seq_len] = self.dropout(torch.relu(self.mov_linear(src)))
        upd_encoder_outputs = self.dropout(torch.relu(self.upd_linear(src)))
        ins_encoder_outputs = self.dropout(torch.relu(self.ins_linear(src)))

        # Cat padded tensors
        for i, batch_length in enumerate(src_lengths):
            combined[batch_length: 2 * batch_length, i] = upd_encoder_outputs[:batch_length, i]
            combined[2 * batch_length: Constants.NUM_OF_OPS * batch_length, i] = ins_encoder_outputs[:batch_length, i]
        # (NUM_OF_CTRL_TOKENS + NUM_OF_OPS * encoded_seq_len, batch_size, h_dim)
        output = torch.cat([pad, eos, combined], dim=0)
        output_length = Constants.NUM_OF_CTRL_TOKENS + Constants.NUM_OF_OPS * src_lengths
        output = pack_padded_sequence(output, output_length, enforce_sorted=False)
        return output


