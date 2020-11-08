import torch.nn as nn
from Models.PathEncoder import PathEncoder
from Models.SequenceEncoder import SequenceEncoder
from Models.Transformer import Transformer
from Models.OperationMix import OperationMix
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, pack_sequence
import torch


class Encoder(nn.Module):
    def __init__(self, path_vocab_size, src_tgt_vocab_size, position_vocab_size, in_dim, h_dim, num_layers, dropout, device, ctx_mode, padding_idx=0):
        super(Encoder, self).__init__()
        self.path_encoder = PathEncoder(path_vocab_size, src_tgt_vocab_size, position_vocab_size, in_dim, h_dim, num_layers, dropout, padding_idx)
        if ctx_mode == 'lstm':
            self.ctx_encoder = SequenceEncoder(h_dim, h_dim, num_layers, dropout)
        elif ctx_mode == 'transformer':
            self.ctx_encoder = Transformer(in_dim=h_dim, dropout=dropout, device=device)
        else:
            self.ctx_encoder = None
        self.split_token = nn.Parameter(torch.Tensor(1, h_dim))
        torch.nn.init.xavier_uniform_(self.split_token.data)
        self.operation_mix = OperationMix(h_dim, dropout)
        self.h_dim = h_dim
        self.num_layers = num_layers
        self.device = device

    def create_split_tokens(self, batch_size):
        return self.split_token.repeat(batch_size, 1).unsqueeze(dim=0)

    def forward(self, packed_srcs, packed_srcs_positions, packed_tgts, packed_tgts_positions, packed_paths, packed_paths_positions, focus_num_of_paths, before_ctx_num_of_paths, after_ctx_num_of_paths):
        """
        :param packed_srcs: PackedSequence of shape (src_length, batch_1)
        :param packed_srcs_positions: PackedSequence of shape (src_length, batch_1)
        :param packed_tgts: PackedSequence of shape (tgt_length, batch_1)
        :param packed_tgts_positions: PackedSequence of shape (tgt_length, batch_1)
        :param packed_paths: PackedSequence of shape (path_length, batch_1)
        :param packed_paths_positions: PackedSequence of shape (path_length, batch_1)
        :return: path_encoded, ctx: of shape (batch, h_dim)
        """
        batch_size = len(focus_num_of_paths)

        # (num_all_paths, h_dim)
        encoded_path = self.path_encoder(packed_srcs, packed_srcs_positions, packed_tgts, packed_tgts_positions,
                                         packed_paths, packed_paths_positions)

        num_of_paths = focus_num_of_paths + before_ctx_num_of_paths + after_ctx_num_of_paths
        encoded_path_list = torch.split(encoded_path, num_of_paths, dim=0)

        encoded_focus_path_list = encoded_path_list[:batch_size]
        before_ctx_encoded_path_list = encoded_path_list[batch_size: 2 * batch_size]
        after_ctx_encoded_path_list = encoded_path_list[-batch_size:]

        # h_list = list(map(lambda t: torch.mean(torch.cat(t, dim=0), dim=0).unsqueeze(dim=0).unsqueeze(dim=0), zip(before_ctx_encoded_path_list, encoded_focus_path_list, after_ctx_encoded_path_list)))
        #
        # # (num_layers, batch_size, h_dim)
        # h = torch.cat(h_list, dim=1).repeat(self.num_layers, 1, 1)

        packed_encoded_path = pack_sequence(encoded_focus_path_list, enforce_sorted=False)

        # TODO: Consider mixing the paths with Transformer before the operation_mix

        packed_mixed_encoded_path = self.operation_mix(packed_encoded_path)

        split_tokens = [self.split_token] * batch_size

        # real_batch_size * ((before_ctx_num_of_paths, h_dim),(1, h_dim),(after_ctx_num_of_paths, h_dim))
        ctx_encoded_path_list = list(map(lambda t: torch.cat(t, dim=0), zip(before_ctx_encoded_path_list, split_tokens, after_ctx_encoded_path_list)))
        ctx_encoded_path_packed = pack_sequence(ctx_encoded_path_list, enforce_sorted=False)
        packed_encoded_ctx = ctx_encoded_path_packed
        if self.ctx_encoder is not None:
            # (real_batch_size, num_of_paths, num_directions * h_dim)
            packed_encoded_ctx = self.ctx_encoder(ctx_encoded_path_packed)

        padded_encoded_path, encoded_lengths = pad_packed_sequence(packed_encoded_path, batch_first=True)
        padded_encoded_ctx_path, encoded_ctx_lengths = pad_packed_sequence(packed_encoded_ctx, batch_first=True)
        lengths = encoded_lengths + encoded_ctx_lengths
        h = (padded_encoded_path.sum(dim=1) + padded_encoded_ctx_path.sum(dim=1)) / lengths.to(self.device).view(-1, 1)
        h = h.unsqueeze(dim=0).repeat(self.num_layers, 1, 1)
        return packed_mixed_encoded_path, packed_encoded_ctx, h

