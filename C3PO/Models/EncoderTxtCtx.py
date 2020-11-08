import torch.nn as nn
from Models.PathEncoder import PathEncoder
from Models.SequenceEncoder import SequenceEncoder
from Models.Embedding import Embedding
from Models.OperationMix import OperationMix
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
import torch


class EncoderTxtCtx(nn.Module):
    def __init__(self, path_vocab_size, src_tgt_vocab_size, position_vocab_size, ctx_vocab_size, in_dim, h_dim, num_layers, dropout, device, padding_idx=0):
        super(EncoderTxtCtx, self).__init__()
        self.path_encoder = PathEncoder(path_vocab_size, src_tgt_vocab_size, position_vocab_size, in_dim, h_dim, num_layers, dropout, padding_idx)
        self.ctx_emb = Embedding(ctx_vocab_size, in_dim, padding_idx)
        self.ctx_encoder = SequenceEncoder(in_dim, h_dim, num_layers, dropout)
        self.operation_mix = OperationMix(h_dim, dropout)
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device

    def create_split_tokens(self, batch_size):
        return self.split_token.repeat(batch_size, 1).unsqueeze(dim=0)

    def forward(self, packed_srcs, packed_srcs_positions, packed_tgts, packed_tgts_positions, packed_paths, packed_paths_positions, packed_ctx, focus_num_of_paths):
        """
        :param packed_srcs: PackedSequence of shape (src_length, batch_1)
        :param packed_srcs_positions: PackedSequence of shape (src_length, batch_1)
        :param packed_tgts: PackedSequence of shape (tgt_length, batch_1)
        :param packed_tgts_positions: PackedSequence of shape (tgt_length, batch_1)
        :param packed_paths: PackedSequence of shape (path_length, batch_1)
        :param packed_paths_positions: PackedSequence of shape (path_length, batch_1)
        :return: path_encoded, ctx: of shape (batch, h_dim)
        """

        # (num_all_paths, h_dim)
        encoded_path = self.path_encoder(packed_srcs, packed_srcs_positions, packed_tgts, packed_tgts_positions,
                                         packed_paths, packed_paths_positions)

        encoded_path_list = torch.split(encoded_path, focus_num_of_paths, dim=0)

        packed_encoded_path = pack_sequence(encoded_path_list, enforce_sorted=False)

        packed_mixed_encoded_path = self.operation_mix(packed_encoded_path)

        ctx, ctx_lengths = pad_packed_sequence(packed_ctx, batch_first=True)
        #(batch_size, seq_len, in_dim)
        ctx_in = self.ctx_emb(ctx)
        packed_ctx_in = pack_padded_sequence(ctx_in, ctx_lengths, enforce_sorted=False, batch_first=True)

        # (batch_2, h_dim)
        packed_encoded_ctx = self.ctx_encoder(packed_ctx_in)

        padded_encoded_path, encoded_lengths = pad_packed_sequence(packed_encoded_path, batch_first=True)
        padded_encoded_ctx_path, encoded_ctx_lengths = pad_packed_sequence(packed_encoded_ctx, batch_first=True)
        lengths = encoded_lengths + encoded_ctx_lengths
        h = (padded_encoded_path.sum(dim=1) + padded_encoded_ctx_path.sum(dim=1)) / lengths.to(self.device).view(-1, 1)
        h = h.unsqueeze(dim=0).repeat(self.num_layers, 1, 1)
        return packed_mixed_encoded_path, packed_encoded_ctx, h

