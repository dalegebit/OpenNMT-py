from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules


def sample_gumbel(type_template, size, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = type_template.data.new(*size).uniform_()
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + Variable(sample_gumbel(logits, logits.size()))
    return torch.nn.functional.softmax(y / temperature)


def gumbel_softmax(logits, temperature, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
        logits: [batch_size, n_class] unnormalized log-probs
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature)
    y_max_score, y_max_ind = y.max(1)
    if hard:
        y_hard = y.eq(y_max_score.unsqueeze(1)).type_as(y)
        y = (y_hard - y).detach() + y
    return y, y_max_ind


class Sampler(object):
    """
    Very efficient sampler, can sample multiple samples for multiple
    (as many as the gpu memory allows) inputs once
    """

    def __init__(self, model, opt, scorer=None):
        super(Sampler, self).__init__()
        self.model = model
        self.generator = model.generator
        self.scorer = scorer
        self.use_gpu = len(opt.gpus) > 0
        self.multi_gpu = len(opt.gpus) > 1
        self.n_gram = opt.decoder_ngram
        # self.max_len = opt.max_sent_length
        self.max_num_seqs = 120
        self.max_len = 50
        self.max_len_ratio = 1.5
        # self.n_sample = opt.rl_sample_size
        self.alpha = 5e-3
        self.temperature = 1.0
        self.gumbel = False

    def sample(self, n_sample, srcs, tgts, lengths, start_pos=1,
               enc_states=None, contexts=None, embs=None,
               largest_len=None, is_eval=False):

        ###################################################################
        # 1. Encode enc_states if not given
        if enc_states is None or contexts is None:
            enc_states, contexts, embs = self.model.encoder(
                srcs, lengths=lengths, largest_len=largest_len
            )
            enc_states = self.model.init_decoder_state(contexts, enc_states)
            start_pos = 1
            
        seqL, batch_size, rnn_size = contexts.size()
        pad_masks = srcs[:, :, 0].data.eq(onmt.Constants.PAD).t()

        def mask(pad):
            self.model.decoder.attn.applyMask(pad)

        # Returns
        all_seqs = []
        all_log_probs = []
        all_probs = []  # softmax(log_prob)
        all_bleus = []

        n = self.max_num_seqs // n_sample  # batch per sample

        ###################################################################
        # 2. For each encoded source, decode multiple target seqs
        for enc_state, context, emb, pad, src, tgt in \
            zip(enc_states.split(n, 1, 0), contexts.split(n, 1),
                embs.detach().split(n, 1), pad_masks.split(n, 0),
                srcs.split(n, 1), tgts.split(n, 1)):
            # a. Prepare anything for every sequence to sample
            #  if only 1 source seq is given, we can simply use expand
            if n == 1:
                enc_state.expandAsBatch_(n_sample)
                context = context.expand(seqL, n_sample, rnn_size)
                emb = emb.expand(seqL, n_sample, rnn_size)
                pad = pad.expand(n_sample, seqL)
                tgt_t = tgt.t().expand(n_sample, tgt.size(0))
                src = src.expand(src.size(0), n_sample, src.size(2))
            #  if more than 1 source seq is given, we must use repeat
            else:
                def _repeat(t, dim, b):
                    shape = [1] * (len(t.size())+1)
                    shape[dim+1] = b
                    view_shape = list(t.size())
                    view_shape[dim] *= b
                    return t.unsqueeze(dim+1).repeat(*shape).view(*view_shape)

                enc_state.repeatAsBatch_(n_sample, _repeat)
                context = _repeat(context, 1, n_sample)
                emb = _repeat(emb, 1, n_sample)
                pad = _repeat(pad, 0, n_sample)
                tgt_t = _repeat(tgt.t(), 0, n_sample)
                src = _repeat(src, 1, n_sample)

            max_len = int(self.max_len_ratio * lengths.data.max())+2

            dec_states = enc_state

            n_seqs = src.size(1)
            result_seqs = [[onmt.Constants.BOS] for _ in range(n_seqs)]
            accum_scores = [Variable(contexts.data.new([0]), volatile=is_eval)
                            for _ in range(n_seqs)]
            cur_seqs = result_seqs
            cur_accum_scores = accum_scores

            input = Variable(tgt_t.data.new(1, n_seqs).
                             fill_(onmt.Constants.BOS), volatile=is_eval)
            n_old_active = n_seqs
            mask(pad.unsqueeze(0))

            # b. Begin Sampling
            for i in range(start_pos, max_len):
                dec_out, dec_states, attn = self.model.decoder(input,
                                                               src,
                                                               context,
                                                               dec_states,
                                                               emb)

                dec_out = dec_out.squeeze(0)
                out = self.generator.forward(dec_out)
                if self.gumbel:
                    onehots, pred_t = gumbel_softmax(out, self.temperature)
                    pred_t = pred_t.unsqueeze(1)
                    onehots = onehots.unsqueeze(0)
                else:
                    pred_t = out.exp().multinomial(1).detach()

                active = Variable(pred_t.data.ne(onmt.Constants.EOS)
                                             .squeeze().nonzero().squeeze(),
                                  volatile=is_eval)

                score_t = out.gather(1, pred_t).squeeze(-1)
                tokens = pred_t.squeeze().data

                # Update scores and sequences
                for i in range(len(cur_accum_scores)):
                    cur_accum_scores[i] += score_t[i]
                    cur_seqs[i].append(tokens[i])

                # If none is active, then stop
                if len(active.size()) == 0 or active.size(0) == 0:
                    break

                if active.size(0) == n_old_active:
                    input = onehots if self.gumbel else pred_t.t()
                    continue

                n_old_active = active.size(0)

                # If some seqs are dead, update envionments for active ones
                new_accum_scores = []
                new_seqs = []
                for b in list(active.data):
                    new_accum_scores.append(cur_accum_scores[b])
                    new_seqs.append(cur_seqs[b])
                cur_accum_scores = new_accum_scores
                cur_seqs = new_seqs

                input = onehots.index_select(1, active) \
                    if self.gumbel else pred_t.t().index_select(1, active)
                src = src.index_select(1, active)
                context = context.index_select(1, active)
                emb = emb.index_select(1, active)
                pad = pad.index_select(0, active.data)
                mask(pad.unsqueeze(0))
                dec_states.activeUpdate_(active)
                if self.n_gram:
                    for i, h in enumerate(self.model.decoder.rnn.histories):
                        self.model.decoder.rnn.histories[i] = \
                            h.index_select(1, active)

            # c. Calculate reward for this batch
            if self.n_gram:
                self.model.decoder.rnn.clear_histories()
            accum_scores = torch.cat(accum_scores, 0)
            prob = nn.functional.softmax(
                        accum_scores.view(-1, n_sample)
                   ).view(-1)
            if self.scorer:
                bleu = self.scorer.score(result_seqs, tgt_t)
                bleu = Variable(accum_scores.data.new(bleu), volatile=is_eval)
                all_bleus.append(bleu)
            all_seqs.extend(result_seqs)
            all_log_probs.append(accum_scores)
            all_probs.append(prob)

        self.model.decoder.attn.removeMask()
        all_log_probs = torch.cat(all_log_probs, 0)
        all_probs = torch.cat(all_probs, 0)
        all_bleus = torch.cat(all_bleus, 0)
        return all_seqs, all_log_probs, all_probs, all_bleus
