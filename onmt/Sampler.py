from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
from onmt.modules.Util import aeq

import copy


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


def expand_variables(n_sample, n_src_per_batch, enc_state, context,
                     emb, src, tgt, input):
    """
    Expand the variables in the batch dimension, which will be provided
    for each sequence that is going to decode
    """
    # if only 1 source seq is given, we can simply use expand
    if n_src_per_batch == 1:
        enc_state.expandAsBatch_(n_sample)
        context = context.expand(context.size(0), n_sample, context.size(2))
        emb = emb.expand(emb.size(0), n_sample, emb.size(2))
        src = src.expand(src.size(0), n_sample, src.size(2))
        if tgt is not None:
            tgt = tgt.expand(tgt.size(0), n_sample)
        input = input.expand(input.size(0), n_sample)

    #  if more than 1 source seq is given, we must use repeat
    #  e.g.                                 [[1,2,3,4],
    #        [[1,2,3,4],   repeat twice      [1,2,3,4],
    #        [5,6,7,8]]      ---->           [5,6,7,8],
    #                                        [5,6,7,8]]
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
        src = _repeat(src, 1, n_sample)
        if tgt is not None:
            tgt = _repeat(tgt, 1, n_sample)
        input = _repeat(input, 1, n_sample)

    return enc_state, context, emb, src, tgt, input


def update_paths(active_idx, cur_scores, cur_seqs, cur_states,
                 return_state):
    """
    Select out the variables whose sequence is alive
    """
    # If some seqs are dead, update envionments for active ones
    # for cur_seqs:
    #       [[BOS, ..., EOS], (dead)
    #        [BOS, ..., xx],  (alive)  -->  [[BOS, ..., xx]]
    #        [BOS, ..., EOS]  (dead)]
    new_scores = []
    new_seqs = []
    new_states = []
    for b in active_idx:
        new_scores.append(cur_scores[b])
        new_seqs.append(cur_seqs[b])
        if return_state:
            new_states.append(cur_states[b])
    return new_scores, new_seqs, new_states


class Sampler(object):
    """
    Very efficient sampler, can decode multiple sequences for multiple
    (as many as the gpu memory allows) inputs once
    """

    def __init__(self, model, opt):
        super(Sampler, self).__init__()
        self.model = model
        self.generator = model.generator
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

    def sample(self, n_sample, srcs, lengths=None, start_pos=1, enc_states=None,
               contexts=None, embs=None, inputs=None, prev_seqs=None, align=None,
               tgts=None, scorer=None, is_eval=False, return_state=False):

        # Encode enc_states if not given
        if enc_states is None or contexts is None:
            enc_states, contexts, embs = self.model.encoder(
                srcs, lengths=lengths, align=align
            )
            enc_states = self.model.init_decoder_state(contexts, enc_states)

        seqL, batch_size, rnn_size = contexts.size()

        def mask(pad):
            self.model.decoder.attn.applyMask(pad)

        def nones():
            while True:
                yield None

        def split_list(l, n):
            i = 0
            while i < len(l):
                yield l[i:i+n]
                i += n

        # Returns
        all_seqs = []
        all_log_probs = []
        all_metrics = []
        all_states = []  # dec_states at every time step
        all_contexts = []
        all_embs = []
        all_srcs = []
        all_tgts = []

        n = self.max_num_seqs // n_sample  # batch per sample

        # For each encoded source, decode multiple target seqs
        for enc_state, context, emb, src, tgt, input, prev_seq in \
            zip(enc_states.split(n, 1, 0),
                contexts.split(n, 1),
                embs.detach().split(n, 1) if embs is not None else nones(),
                srcs.split(n, 1),
                tgts[1:].split(n, 1) if tgts is not None else nones(),
                inputs.split(n, 1) if inputs is not None else nones(),
                split_list(prev_seqs, n) if prev_seqs is not None else nones()):
            ##################################################
            # a. Prepare anything for every sequence to sample
            ##################################################
            if input is None:
                input = Variable(src.data.new(1, src.size(1)).
                                 fill_(onmt.Constants.BOS), volatile=is_eval)
            aeq(input.size(0), 1)

            enc_state, context, emb, src, tgt, input = \
                expand_variables(n_sample, n, enc_state, context, emb,
                                 src, tgt, input)
            tgt_t = tgt.t()

            if tgt.size(1) != src.size(1):
                print(tgts.size(), srcs.size())
                print(tgt.size(), src.size())
                import pdb; pdb.set_trace()

            max_len = int(self.max_len_ratio * lengths.data.max())+2

            dec_states = enc_state
            pad = src[:, :, 0].data.eq(onmt.Constants.PAD).t()

            n_seqs = src.size(1)
            if prev_seqs is not None:
                seqs = []
                for seq in prev_seq:
                    seqs.extend([copy.deepcopy(seq) for _ in range(n_sample)])
            else:
                seqs = [[] for _ in range(n_seqs)]
            scores = [[] for _ in range(n_seqs)]
            states = [[] for _ in range(n_seqs)]

            cur_seqs = seqs
            cur_scores = scores
            cur_states = states

            if return_state:
                all_contexts.extend(context.split(1, 1))
                if emb is not None:
                    all_embs.extend(context.split(1, 1))
                all_srcs.extend(src.split(1, 1))
                if tgt is not None:
                    all_tgts.extend(tgt.split(1, 1))

            n_old_active = n_seqs
            mask(pad.unsqueeze(0))
            ###################
            # b. Begin Sampling
            ###################
            for i in range(start_pos, max_len):

                dec_out, dec_states, attn = \
                    self.model.decoder(input, src, context, dec_states, emb)

                dec_out = dec_out.squeeze(0)
                out = self.generator.forward(dec_out)
                if self.gumbel:
                    onehots, pred = gumbel_softmax(out, self.temperature)
                    pred = pred.unsqueeze(1)
                    onehots = onehots.unsqueeze(0)
                else:
                    pred = out.exp().multinomial(1).detach()

                active = Variable(pred.data.ne(onmt.Constants.EOS)\
                                  .squeeze().nonzero().squeeze(), volatile=is_eval)

                log_probs = out.gather(1, pred).squeeze(-1)
                tokens = pred.squeeze().data

                # Update scores and sequences
                for j, (score, token) in enumerate(zip(log_probs, tokens)):
                    cur_scores[j].append(score)
                    cur_seqs[j].append(token)
                if return_state:
                    for j, state in enumerate(dec_states.split(1, 1, 0)):
                        cur_states[j].append(state)

                # If none is active, then stop
                if len(active.size()) == 0 or active.size(0) == 0:
                    break

                if active.size(0) == n_old_active:
                    input = onehots if self.gumbel else pred.t()
                    continue

                n_old_active = active.size(0)

                cur_scores, cur_seqs, cur_states = \
                    update_paths(list(active.data), cur_scores,
                                 cur_seqs, cur_states, return_state)

                # Select out the variables whose sequence is alive
                input = onehots.index_select(1, active) \
                    if self.gumbel else pred.t().index_select(1, active)
                src = src.index_select(1, active)
                context = context.index_select(1, active)
                emb = emb.index_select(1, active)
                pad = pad.index_select(0, active.data)
                mask(pad.unsqueeze(0))
                dec_states.activeUpdate_(active)
                if self.n_gram:
                    for j, h in enumerate(self.model.decoder.rnn.histories):
                        self.model.decoder.rnn.histories[j] = \
                            h.index_select(1, active)
            ####################################
            # c. Calculate reward for this batch
            ####################################
            if self.n_gram:
                self.model.decoder.rnn.clear_histories()
            if scorer and tgt_t is not None:
                # print(len(seqs), tgt_t.size())
                metrics = scorer.score(seqs, tgt_t)
                all_metrics.extend(metrics)
            all_seqs.extend(seqs)
            all_log_probs.extend(scores)
            if return_state:
                all_states.extend(states)

        self.model.decoder.attn.removeMask()

        return {'seqs': all_seqs,
                'log_probs': all_log_probs,
                'metrics': all_metrics,
                'states': all_states,
                'contexts': all_contexts,
                'embs': all_embs,
                'src': all_srcs,
                'tgt': all_tgts}
