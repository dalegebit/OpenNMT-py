import torch
import torch.nn as nn
from torch.autograd import Variable
import onmt
import onmt.modules
import torch.multiprocessing as mp
from onmt.Loss import RLStatistics
import math


def accum_shared_grads(model, shared_model, lock):
    with lock:
        for param, shared_param in zip(model.parameters(),
                                       shared_model.parameters()):
            if shared_param.grad is None:
                shared_param._grad = param.grad.cuda(
                    shared_param.data.get_device())
            else:
                shared_param._grad.data += param.grad.data.cuda(
                    shared_param.data.get_device())


def prepare_input_for_t(t, cur_seqs, cur_states, alive_idx, gpuid):
    states = onmt.Models.RNNDecoderState.cat([cur_states[i][t]
                                              for i in alive_idx])
    inputs = Variable(
        torch.cuda.LongTensor([[cur_seqs[i][t] for i in alive_idx]],
                              device=gpuid))
    prev_seqs = [cur_seqs[i][:t+1] for i in alive_idx]
    return inputs, states, prev_seqs


def check_alive(t, seqs, old_alive_idx):
    alive_idx = []
    for idx in old_alive_idx:
        if t >= len(seqs[idx])-1 or seqs[idx][t] == onmt.Constants.EOS:
            continue
        alive_idx.append(idx)
    return alive_idx


def update_alive(contexts, embs, src, tgt, alive_idx):
    contexts = torch.cat([contexts[j] for j in alive_idx], 1)
    embs = torch.cat([embs[j] for j in alive_idx], 1)
    src = torch.cat([src[j] for j in alive_idx], 1)
    tgt = torch.cat([tgt[j] for j in alive_idx], 1)
    return contexts, embs, src, tgt


def get_dummies(log_probs, states, contexts):
    dummy_log_probs = [[Variable(lp.data, requires_grad=True)
                        for lp in log_prob] for log_prob in log_probs]
    dummy_states = [[s.getDummy() for s in state] for state in states]
    dummy_contexts = [Variable(context.data, requires_grad=True)
                      for context in contexts]
    return dummy_log_probs, dummy_states, dummy_contexts


def get_loss(log_probs, metrics, n_rollout, gpuid, base=0.5):
    metrics = Variable(torch.cuda.FloatTensor(metrics, device=gpuid))
    metrics = metrics.view(-1, n_rollout)
    reward = metrics.mean(1) - base
    log_probs = torch.cat(log_probs, 0)
    return -(log_probs * reward).sum()


def collect_grad(dummy_log_probs, dummy_states, dummy_contexts,
                 log_probs, states, contexts):
    dummies_grad = []
    real_variables = []

    def _collect_grad(dummies, variables):
        for dummy, variable in zip(dummies, variables):
            if dummy is None:
                continue
            if isinstance(dummy, list):
                _collect_grad(dummy, variable)
            elif isinstance(dummy, onmt.Models.RNNDecoderState):
                _collect_grad(list(dummy.all) + [dummy.coverage],
                              list(variable.all) + [variable.coverage])
            else:
                if dummy.grad is not None:
                    dummies_grad.append(dummy.grad.data)
                    real_variables.append(variable)

    _collect_grad(dummy_log_probs, log_probs)
    _collect_grad(dummy_states, states)
    _collect_grad(dummy_contexts, contexts)
    return dummies_grad, real_variables


def train_agent(rank, opt, dicts, queue, out_queue, model, locks, shard_size):
    local_model, local_gen = onmt.Models.build_model(opt, dicts)
    local_model.generator = local_gen
    local_model.cuda(rank)
    sampler = onmt.Sampler(local_model, opt)
    scorer = onmt.Loss.BleuScore()
    local_statistics = RLStatistics()
    while True:
        batch = queue.get()
        batch.cuda(rank)
        local_model.load_state_dict(model.state_dict())
        local_model.zero_grad()
        for b in batch.xsplit(5):
            dicts = sampler.sample(opt.rl_sample_num, b.src, b.lengths,
                                   tgts=b.tgt, return_state=True)
            all_seqs, all_log_probs, all_dec_states, all_contexts, all_embs, \
                all_src, all_tgt = (dicts['seqs'], dicts['log_probs'],
                                    dicts['states'], dicts['contexts'],
                                    dicts['embs'], dicts['src'],
                                    dicts['tgt'])
            i = 0
            sect_size = max(1, (opt.max_samples_per_optim // opt.rollout)
                            // opt.rl_sample_num * opt.rl_sample_num)

            while i < len(all_seqs):
                selected_idx = list(range(i, min(i+sect_size, len(all_seqs))))
                (cur_seqs, cur_log_probs, cur_states, cur_contexts,
                 cur_embs, cur_src, cur_tgt) = \
                    zip(*[(all_seqs[j], all_log_probs[j], all_dec_states[j],
                           all_contexts[j], all_embs[j], all_src[j],
                           all_tgt[j]) for j in selected_idx])
                # Collect gradients on dummy variables to avoid the graph being
                # free after one backpropagation
                dummy_log_probs, dummy_states, dummy_contexts = \
                    get_dummies(cur_log_probs, cur_states, cur_contexts)
                # Concatenate contexts and embs
                contexts = torch.cat(dummy_contexts, 1)
                embs = torch.cat(cur_embs, 1)
                src = torch.cat(cur_src, 1)
                tgt = torch.cat(cur_tgt, 1)
                alive_idx = list(range(len(selected_idx)))
                t = 0
                while len(alive_idx) > 0:
                    inputs, states, prev_seqs = \
                        prepare_input_for_t(t, cur_seqs, dummy_states,
                                            alive_idx, rank)
                    # print(inputs.size(), contexts.size())
                    dicts = sampler.sample(opt.rollout, src, b.lengths,
                                           start_pos=t+1, inputs=inputs,
                                           prev_seqs=prev_seqs,
                                           contexts=contexts, embs=embs,
                                           enc_states=states,
                                           tgts=tgt, scorer=scorer)

                    loss = get_loss([dummy_log_probs[idx][t]
                                     for idx in alive_idx],
                                    dicts['metrics'], opt.rollout, rank)
                    loss.div(batch.batchSize * opt.rl_sample_num).backward()

                    # Calculate statistics
                    # print([len(s) for s in dicts['seqs']])
                    first_moment = sum([m for m in dicts['metrics']])
                    second_moment = sum([pow(m, 2) for m in dicts['metrics']])
                    n_src_words = sum([len(seq) for seq in dicts['seqs']]) - \
                        sum([len(seq) for seq in prev_seqs])
                    n_words = opt.rollout * opt.rl_sample_num * \
                        b.tgt.ne(onmt.Constants.PAD).data.sum()
                    local_statistics.update(
                        RLStatistics(first_moment, second_moment,
                                     n_words, len(dicts['seqs']),
                                     n_src_words))

                    new_alive_idx = check_alive(t, cur_seqs, alive_idx)
                    if len(new_alive_idx) == 0:
                        break
                    if len(alive_idx) != len(new_alive_idx):
                        contexts, embs, src, tgt = \
                            update_alive(cur_contexts, cur_embs, cur_src,
                                         cur_tgt, new_alive_idx)

                    alive_idx = new_alive_idx
                    t += 1

                grads, variables = collect_grad(dummy_log_probs, dummy_states,
                                                dummy_contexts, cur_log_probs,
                                                cur_states, cur_contexts)
                # Accumulate gradients
                # print(i)
                torch.autograd.backward(variables, grads)
                i += sect_size

        # Synchronize gradients
        accum_shared_grads(local_model.encoder, model.encoder,
                           locks['encoder'])
        accum_shared_grads(local_model.decoder, model.decoder,
                           locks['decoder'])
        accum_shared_grads(local_gen, model.generator, locks['generator'])
        out_queue.put(local_statistics)
        local_statistics = RLStatistics()


class RLBatchTrainer(object):
    """
    It is actually not a trainer, I just don't know how to call it.
    It simply calculates the gradient given a batch and does not
    perform optimizing step.
    """
    def __init__(self, model, dicts, opt):
        mp.set_start_method('spawn')
        self.model = model
        self.generator = model.generator
        self.model.share_memory()
        self.opt = opt
        self.dicts = dicts  # for initializing embeddings
        self.gpus = opt.gpus
        self.shard_size = opt.max_samples_per_optim // opt.rl_sample_num
        self.sample_num = opt.rl_sample_num
        self.queue = mp.SimpleQueue()
        self.recv_queue = mp.SimpleQueue()
        self.locks = {'encoder': mp.Lock(), 'decoder': mp.Lock(),
                      'generator': mp.Lock()}
        self.agents = []

        for gpu in self.gpus:
            p = mp.Process(target=train_agent,
                           args=(gpu, self.opt, self.dicts,
                                 self.queue, self.recv_queue,
                                 self.model, self.locks, self.shard_size))
            p.start()
            self.agents.append(p)

    def get_batch_grad(self, batch):
        statistics = RLStatistics()
        split_size = int(math.ceil(batch.batchSize / len(self.gpus)))
        self.model.zero_grad()
        for minibatch in batch.xsplit(split_size):
            self.queue.put(minibatch)
        for _ in self.gpus:
            statistics.update(self.recv_queue.get())
        return statistics

    def eval_batch(self, batch):
        statistics = RLStatistics()
        sampler = onmt.Sampler(self.model, self.opt)
        scorer = onmt.Loss.BleuScore()
        self.model.eval()
        batch.cuda(self.gpus[0])
        for b in batch.xsplit(self.shard_size):  # mini_size = 6
            dicts = sampler.sample(self.sample_num, b.src, b.tgt, b.lengths,
                                   align=b.lengths.data.max(),
                                   is_eval=True, scorer=scorer)
            bleus = dicts['metrics']
            first_moment = sum([m for m in bleus])
            second_moment = sum([pow(m, 2) for m in bleus])
            n_words = self.sample_num * b.tgt.ne(onmt.Constants.PAD).data.sum()
            statistics.update(
                RLStatistics(first_moment, second_moment, n_words,
                             self.shard_size*self.sample_num))
        self.model.train()
        return statistics
