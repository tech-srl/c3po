import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import Constants


class Attention(nn.Module):
    def __init__(self, h_dim, attention_type='general'):
        super(Attention, self).__init__()

        if attention_type not in ['dot', 'general']:
            raise ValueError('Invalid attention type selected.')

        self.attention_type = attention_type
        if self.attention_type == 'general':
            self.linear_in = nn.Linear(h_dim, h_dim, bias=False)

        self.linear_out = nn.Linear(h_dim * 2, h_dim, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, context):
        """
        Args:
            query (:class:`torch.PackedSequence` [batch size, output length, h_dim]): Sequence of
                queries to query the context.
            context (:class:`torch.PackedSequence` [batch size, query length, h_dim]): Data
                overwhich to apply the attention mechanism.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, h_dim]):
              Tensor containing the attended features.
            * **scores** (:class:`torch.FloatTensor` [batch size, 1, query length * output length]):
              Tensor containing attention weights.
        """
        # (batch_size, query_length, h_dim)
        context, context_lengths = pad_packed_sequence(context, batch_first=True)
        # (batch_size, out_length, h_dim)
        query, query_lengths = pad_packed_sequence(query, batch_first=True)

        query_len = context.size(1)
        batch_size, output_len, h_dim = query.size()
        if self.attention_type == "general":
            query = query.view(batch_size * output_len, h_dim)
            query = self.linear_in(query)
            query = query.view(batch_size, output_len, h_dim)
        # (batch_size, h_dim, query_len)
        context = context.transpose(1, 2).contiguous()
        # (batch_size, output_len, h_dim) * (batch_size, h_dim, query_len) ->
        # (batch_size, output_len, query_len)
        attention_scores = torch.bmm(query, context)

        # replace all masked entries with -inf

        packed_attention_scores = pack_padded_sequence(attention_scores, query_lengths, batch_first=True, enforce_sorted=False)
        # (batch_size, output_len, query_len)
        attention_scores, query_lengths = pad_packed_sequence(packed_attention_scores, batch_first=True, padding_value=Constants.NEG_INF)

        attention_scores = attention_scores.transpose(1, 2)
        packed_attention_scores = pack_padded_sequence(attention_scores, context_lengths, batch_first=True, enforce_sorted=False)

        attention_scores, _ = pad_packed_sequence(packed_attention_scores, batch_first=True, padding_value=Constants.NEG_INF)
        attention_scores = attention_scores.transpose(1, 2)

        # Compute weights across every context sequence
        # (batch_size, output_len, query_len)
        attention_weights = self.softmax(attention_scores)

        # (batch_size, query_len, h_dim)
        ##context = context.view(batch_size, query_len, -1)
        context = context.transpose(1, 2).contiguous()
        # (batch_size, output_len, query_len) * (batch_size, query_len, h_dim) ->
        # (batch_size, output_len, h_dim)
        mix = torch.bmm(attention_weights, context)

        # (batch_size, output_len, 2 * h_dim)
        combined = torch.cat((mix, query), dim=2)

        # Apply linear_out on every 2nd dimension of concat
        # output -> (batch_size, output_len, dimensions)
        output = self.linear_out(combined)
        output = torch.tanh(output)

        return output, packed_attention_scores
