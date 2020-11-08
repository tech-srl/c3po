import torch.nn as nn
from Models.Transformer import Transformer
import torch


class NodeEncoder(nn.Module):
    def __init__(self, h_dim):
        super(NodeEncoder, self).__init__()
        self.transformer = Transformer(in_dim=h_dim)

    def forward(self, paths):
        """
        :param paths: of shape (num_of_paths, batch_size, h_dim)
        :return: out: of shape (batch_size, h_dim)
        """
        # (num_of_paths, batch_size, h_dim)
        transformer_output = self.transformer(paths)
        # (batch_size, h_dim)
        output = torch.mean(transformer_output, dim=0, keepdim=False)
        return output

