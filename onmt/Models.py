import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt.modules
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

class Encoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.num_directions = 2 if opt.brnn else 1
        assert opt.rnn_size % self.num_directions == 0
        self.hidden_size = opt.rnn_size // self.num_directions
        input_size = opt.word_vec_size

        super(Encoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = nn.LSTM(input_size, self.hidden_size,
                        num_layers=opt.layers,
                        dropout=opt.dropout,
                        bidirectional=opt.brnn)
        self.attn_use_emb = opt.attn_use_emb

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_enc)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden=None):
        if isinstance(input, tuple):
            lengths = input[1].data.view(-1).tolist() # lengths data is wrapped inside a Variable
            emb = pack(self.word_lut(input[0]), lengths)
        else:
            emb = self.word_lut(input)
        outputs, hidden_t = self.rnn(emb, hidden)
        if isinstance(input, tuple):
            outputs = unpack(outputs)[0]
            if self.attn_use_emb:
                emb = unpack(emb)[0]
        if self.attn_use_emb:
            return hidden_t, outputs, emb
        else:
            return hidden_t, outputs


class StackedLSTM(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout):
        super(StackedLSTM, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, rnn_size))
            input_size = rnn_size

    def forward(self, input, hidden):
        h_0, c_0 = hidden
        h_1, c_1 = [], []
        for i, layer in enumerate(self.layers):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            input = h_1_i
            if i + 1 != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)

        return input, (h_1, c_1)

class StackedLSTMWithMultiAttn(nn.Module):
    def __init__(self, num_layers, input_size, rnn_size, dropout, use_emb=False):
        super(StackedLSTMWithMultiAttn, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.attns = nn.ModuleList()
        self.use_emb = use_emb

        in_size = input_size
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(in_size, rnn_size))
            in_size = rnn_size

        if use_emb:
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalKeyValueAttention(rnn_size))
        else:
            for i in range(num_layers):
                self.attns.append(onmt.modules.GlobalAttention(rnn_size))


    def forward(self, input, hidden, context, embedding=None):
        h_0, c_0 = hidden
        h_1, c_1, a_1 = [], [], []
        for i, (layer, attn) in enumerate(zip(self.layers, self.attns)):
            h_1_i, c_1_i = layer(input, (h_0[i], c_0[i]))
            if self.use_emb:
                _, context_1_i, attnWeight_1_i = attn(h_1_i, context.t(), embedding.t())
            else:
                _, context_1_i, attnWeight_1_i = attn(h_1_i, context.t())
            h_1_i = h_1_i + context_1_i
            input = h_1_i
            if i != self.num_layers:
                input = self.dropout(input)
            h_1 += [h_1_i]
            c_1 += [c_1_i]
            a_1 += [attnWeight_1_i]

        h_1 = torch.stack(h_1)
        c_1 = torch.stack(c_1)
        a_1 = torch.stack(a_1)

        return input, (h_1, c_1), a_1


class Decoder(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(Decoder, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTM(opt.layers, input_size, opt.rnn_size, opt.dropout)
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)

        self.hidden_size = opt.rnn_size

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            output, hidden = self.rnn(emb_t, hidden)
            output, _, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn



class DecoderWithMultiAttn(nn.Module):

    def __init__(self, opt, dicts):
        self.layers = opt.layers
        self.input_feed = opt.input_feed
        input_size = opt.word_vec_size
        if self.input_feed:
            input_size += opt.rnn_size

        super(DecoderWithMultiAttn, self).__init__()
        self.word_lut = nn.Embedding(dicts.size(),
                                  opt.word_vec_size,
                                  padding_idx=onmt.Constants.PAD)
        self.rnn = StackedLSTMWithMultiAttn(opt.layers, input_size, opt.rnn_size, opt.dropout, opt.attn_use_emb) # use emb
        self.attn = onmt.modules.GlobalAttention(opt.rnn_size)
        self.dropout = nn.Dropout(opt.dropout)
        self.attn_use_emb = opt.attn_use_emb

        self.hidden_size = opt.rnn_size

        if opt.pre_word_vecs_enc is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.copy_(pretrained)

    def load_pretrained_vectors(self, opt):
        if opt.pre_word_vecs_dec is not None:
            pretrained = torch.load(opt.pre_word_vecs_dec)
            self.word_lut.weight.data.copy_(pretrained)

    def forward(self, input, hidden, context, init_output, src_emb=None):
        emb = self.word_lut(input)

        # n.b. you can increase performance if you compute W_ih * x for all
        # iterations in parallel, but that's only possible if
        # self.input_feed=False
        outputs = []
        output = init_output
        for emb_t in emb.split(1):
            emb_t = emb_t.squeeze(0)
            if self.input_feed:
                emb_t = torch.cat([emb_t, output], 1)

            if self.attn_use_emb:
                output, hidden, _ = self.rnn(emb_t, hidden, context, src_emb)
            else:
                output, hidden, _ = self.rnn(emb_t, hidden, context)

            output, _, attn = self.attn(output, context.t())
            output = self.dropout(output)
            outputs += [output]

        outputs = torch.stack(outputs)
        return outputs, hidden, attn



class NMTModel(nn.Module):

    def __init__(self, encoder, decoder, attn_use_emb):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.attn_use_emb = attn_use_emb

    def make_init_decoder_output(self, context):
        batch_size = context.size(1)
        h_size = (batch_size, self.decoder.hidden_size)
        return Variable(context.data.new(*h_size).zero_(), requires_grad=False)

    def _fix_enc_hidden(self, h):
        #  the encoder hidden is  (layers*directions) x batch x dim
        #  we need to convert it to layers x batch x (directions*dim)
        if self.encoder.num_directions == 2:
            return h.view(h.size(0) // 2, 2, h.size(1), h.size(2)) \
                    .transpose(1, 2).contiguous() \
                    .view(h.size(0) // 2, h.size(1), h.size(2) * 2)
        else:
            return h

    def forward(self, input):
        src = input[0]
        tgt = input[1][:-1]  # exclude last target from inputs

        if self.attn_use_emb:
            enc_hidden, context, emb = self.encoder(src)
        else:
            enc_hidden, context = self.encoder(src)

        init_output = self.make_init_decoder_output(context)

        enc_hidden = (self._fix_enc_hidden(enc_hidden[0]),
                      self._fix_enc_hidden(enc_hidden[1]))

        pad_mask = src[0].data.eq(onmt.Constants.PAD).t()

        def apply_context_mask(m):
            if isinstance(m, onmt.modules.GlobalAttention) or isinstance(m, onmt.modules.GlobalKeyValueAttention):
                m.applyMask(pad_mask)

        def unbind(variable):
            new_var = variable.data.new(variable.data.size())
            new_var.copy_(variable.data)
            return Variable(new_var, requires_grad=False)

        self.decoder.apply(apply_context_mask)

        if self.attn_use_emb:
            out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output, unbind(emb))
        else:
            out, dec_hidden, _attn = self.decoder(tgt, enc_hidden, context, init_output)


        return out
