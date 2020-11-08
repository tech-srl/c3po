import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, vocab_size, nb_labels, device, num_layer, dropout, emb_dim=10, hidden_dim=10):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(
            2 * emb_dim, hidden_dim // 2, bidirectional=True, batch_first=True, num_layers=num_layer, dropout=dropout
        )
        self.hidden2tag = nn.Linear(hidden_dim, nb_labels)
        self.device = device
        self.num_layer = num_layer

    def init_hidden(self, batch_size):
        # return (
        #     torch.randn(2 * self.num_layer, batch_size, self.hidden_dim // 2, device=self.device),
        #     torch.randn(2 * self.num_layer, batch_size, self.hidden_dim // 2, device=self.device),
        # )
        return (
            torch.zeros(2 * self.num_layer, batch_size, self.hidden_dim // 2, device=self.device),
            torch.zeros(2 * self.num_layer, batch_size, self.hidden_dim // 2, device=self.device),
        )

    def forward(self, batch_of_sentences, encoded_ctx):
        hidden = self.init_hidden(batch_of_sentences.shape[0])
        x = self.emb(batch_of_sentences)
        x = torch.cat([x, encoded_ctx], dim=2)
        x, hidden = self.lstm(x, hidden)

        x = self.hidden2tag(x)
        return x
