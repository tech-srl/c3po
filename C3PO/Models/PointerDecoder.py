import torch.nn as nn
from Models.Attention import Attention
import torch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence


class PointerDecoder(nn.Module):
    def __init__(self, in_dim, h_dim, num_of_layers, device, dropout=0, use_attention=True):
        super(PointerDecoder, self).__init__()
        self.lstm = nn.LSTM(input_size=in_dim,
                            hidden_size=h_dim,
                            num_layers=num_of_layers,
                            dropout=dropout,
                            bidirectional=False)
        self.init_token = nn.Parameter(torch.Tensor(1, in_dim))
        torch.nn.init.xavier_uniform_(self.init_token.data)
        self.use_attention = use_attention
        if self.use_attention:
            self.ctx_attn = Attention(h_dim)
        # self.focus_attn = Attention(h_dim)

        # (2, h_dim)
        self.attn_classifier = Attention(h_dim)
        self.h_dim = h_dim
        self.num_of_layers = num_of_layers
        self.directions = 1
        self.device = device


    def create_h_or_c(self, batch_size):
        return torch.zeros(self.directions * self.num_of_layers, batch_size, self.h_dim, device=self.device)

    def create_initial_inputs(self, batch_size):
        return self.init_token.repeat(batch_size, 1).unsqueeze(dim=0)

    def get_init_token(self):
        return self.init_token

    def forward(self, encoder_outputs, ctx_outputs, batch_size, inputs=None, hc=None):
        """
        :param encoder_outputs: PackedSequence of shape (encoded_seq_len, batch_size, in_dim)
        :param inputs: PackedSequence of shape (seq_len, batch_size, in_dim)
        :param hc: (h_0, c_0) of shape (directions * num_layers, batch, hidden_size)
        :return: attention_weights of shape (batch size, 2 * encoded_seq_len),
                (h_n, c_n) of shape (2 * num_layers, batch, h_dim)
        """
        if hc is None:
            hc = (self.create_h_or_c(batch_size), self.create_h_or_c(batch_size))
        # lstm_out: PackedSequence of shape (seq_len, batch, h_dim)
        # (h_n, c_n): (num_layers, batch, h_dim)
        lstm_out, (h_n, c_n) = self.lstm(inputs, hc)
        # (batch_size, inputs_length , h_dim)
        padded_lstm_out, lstm_out_lengths = pad_packed_sequence(lstm_out, batch_first=True)
        if self.use_attention:
            # (batch, seq_len, h_dim)
            ctx_attention_out, _ = self.ctx_attn(lstm_out, ctx_outputs)
            query_packed = pack_padded_sequence(ctx_attention_out, lstm_out_lengths, batch_first=True, enforce_sorted=False)
            # TODO: Consider adding attention to the encoded paths
            # focus_attention_out, _ = self.focus_attn(query_packed, encoder_outputs)
            # query_packed = pack_padded_sequence(focus_attention_out, lstm_out_lengths, batch_first=True, enforce_sorted=False)
        else:
            query_packed = lstm_out
        # output: (batch_size, h_dim)
        # attention_weights: PackedSequence of shape (batch size, encoded_seq_len, output_len)
        _, attention_scores = self.attn_classifier(query_packed, encoder_outputs)
        return attention_scores, (h_n, c_n)
