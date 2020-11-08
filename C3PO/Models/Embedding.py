import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, vocab_size, h_dim, padding_idx=0):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, h_dim, padding_idx=padding_idx)

    def forward(self, inputs):
        return self.emb(inputs.long())

