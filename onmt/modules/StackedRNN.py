import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules.GlobalAttention


class StackedLSTM(nn.Module):
    """
    Our own implementation of stacked LSTM.
    Needed for the decoder, because we do input feeding.
    """
    def __init__(self, num_layers, input_size, rnn_size, dropout,
                 multi_attn=False, attn_use_emb=False, n_gram=None):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.multi_attn = multi_attn
        self.n_gram = n_gram

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

        if self.multi_attn:
            self.attns = nn.ModuleList()
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalAttention(
                    rnn_size, use_emb=attn_use_emb
                ))

        if self.n_gram:
            self.block_len = rnn_size / self.n_gram
            self.histories = []
            self.ngram_linear = nn.Linear(2*rnn_size, rnn_size)
            # self.ngram_tanh = nn.Tanh()
            self.ngram_activ = nn.ReLU()

    def clear_histories(self):
        self.histories = []

    def forward(self, input, hidden, context=None, emb=None):
        h_0, c_0 = hidden
        if self.n_gram and len(self.histories) == 0:
            self.histories = [Variable(input.data.new(h_0.size()).zero_())
                              for _ in range(self.n_gram)]
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))

            if self.multi_attn and context is not None:
                attn_output, _ = self.attns[i](h_1_i,
                                               context.transpose(0, 1),
                                               emb=emb.transpose(0, 1)
                                               if emb is not None else None)
                h_1_i = h_1_i + attn_output

            if self.n_gram:
                histories = [h[i, :, j*self.block_len:(j+1)*self.block_len]
                             for j, h in enumerate(self.histories)]
                histories = torch.cat(histories, -1)
                h_1_i = self.ngram_activ(
                        self.ngram_linear(
                            torch.cat((h_1_i, histories), -1)
                        ))

            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        if self.n_gram:
            self.histories = [h_1] + \
                             self.histories[0:self.n_gram-1]

        return input, (h_1, c_1)


class StackedGRU(nn.Module):

    def __init__(self, num_layers, input_size, rnn_size, dropout,
                 multi_attn=False, attn_use_emb=False, n_gram=None):
        super(StackedGRU, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.multi_attn = multi_attn

        for i in range(num_layers):
            self.layers.append(nn.GRUCell(input_size, rnn_size))
            input_size = rnn_size

        if self.multi_attn:
            self.attns = nn.ModuleList()
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalAttention(
                    rnn_size, use_emb=attn_use_emb
                ))
        self.n_gram = n_gram
        if self.n_gram:
            self.block_len = rnn_size / self.n_gram
            self.histories = []
            self.ngram_linear = nn.Linear(2*rnn_size, rnn_size)
            # self.ngram_tanh = nn.Tanh()
            self.ngram_activ = nn.ReLU()

    def forward(self, input, hidden, context=None, emb=None):
        if self.n_gram and len(self.histories) == 0:
            self.histories = [Variable(input.data.new(hidden[0].size())
                                       .zero_())
                              for _ in range(self.n_gram)]
        h_1 = []
        for i, layer in enumerate(self.layers):
            h_1_i = layer(input, hidden[0][i])
            if self.multi_attn and context is not None:
                attn_output, _ = self.attns[i](h_1_i,
                                               context.transpose(0, 1),
                                               emb.transpose(0, 1)
                                               if emb is not None else None)
                h_1_i = h_1_i + attn_output

            if self.n_gram:
                histories = [h[i, :, j*self.block_len:(j+1)*self.block_len]
                             for j, h in enumerate(self.histories)]
                histories = torch.cat(histories, -1)
                h_1_i = self.ngram_activ(
                        self.ngram_linear(
                            torch.cat((h_1_i, histories), -1)
                        ))

            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]

        h_1 = torch.stack(h_1)

        if self.n_gram:
            self.histories = [h_1] + \
                             self.histories[0:self.n_gram-1]
        return input, (h_1,)
