# from torch import nn
# from Models.LSTM import LSTM
# from torchcrf import CRF
#
#
# class BiLSTM_CRF_V2(nn.Module):
#     def __init__(self, vocab_size, nb_labels, emb_dim, hidden_dim, bos_idx, eos_idx, pad_idx, num_lstm_layers, device):
#         super().__init__()
#         self.lstm = LSTM(
#             vocab_size, nb_labels, device=device, emb_dim=emb_dim, hidden_dim=hidden_dim, num_layer=num_lstm_layers
#         )
#         self.crf = CRF(
#             nb_labels,
#             batch_first=True
#         )
#
#     def forward(self, x, mask=None):
#         emissions = self.lstm(x)
#         path = self.crf.decode(emissions, mask=mask.bool())
#         return None, path
#
#     def loss(self, x, y, mask=None):
#         emissions = self.lstm(x)
#         nll = -self.crf(emissions, y, mask=mask.bool())
#         return nll
