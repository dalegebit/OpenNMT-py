from __future__ import division
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


def train_agent(rank, opt, dicts, queue, out_queue, model, locks):
    local_model, local_gen = onmt.Models.build_model(opt, dicts)
    local_model.generator = local_gen
    local_model.load_state_dict(model.state_dict())
    local_model.cuda(rank)
    sampler = onmt.Sampler(local_model, opt, onmt.Loss.BleuScore())
    while True:
        batch = queue.get()
        batch.cuda(rank)
        local_model.zero_grad()
        for b in batch.xsplit(6):  # mini_size = 6
            samples, _, probs, scores = \
                sampler.sample(20, b.src, b.tgt, b.lengths,
                               largest_len=b.lengths.data.max())
            total_loss = (probs * scores).sum() / opt.batch_size
            total_loss.backward()
        accum_shared_grads(local_model.encoder, model.encoder, locks['encoder'])
        accum_shared_grads(local_model.decoder, model.decoder, locks['decoder'])
        accum_shared_grads(local_gen, model.generator, locks['generator'])
        out_queue.put(rank)


class RLBatchTrainer(object):
    def __init__(self, model, dicts, opt):
        mp.set_start_method('spawn')
        self.model = model
        self.generator = model.generator
        self.model.share_memory()
        self.opt = opt
        self.dicts = dicts  # for initializing embeddings
        self.gpus = opt.gpus
        self.queue = mp.SimpleQueue()
        self.recv_queue = mp.SimpleQueue()
        self.locks = {'encoder': mp.Lock(), 'decoder': mp.Lock(),
                      'generator': mp.Lock()}
        self.agents = []

        for gpu in self.gpus:
            p = mp.Process(target=train_agent,
                           args=(gpu, self.opt, self.dicts,
                                 self.queue, self.recv_queue,
                                 self.model, self.locks))
            p.start()
            self.agents.append(p)

    def get_batch_grad(self, batch):
        split_size = int(math.ceil(batch.batchSize / len(self.gpus)))
        for minibatch in batch.xsplit(split_size):
            self.queue.put(minibatch)
        for _ in self.gpus:
            self.recv_queue.get()

    def eval_batch(self, batch):
        rl_statistics = RLStatistics()
        sampler = onmt.Sampler(self.model, self.opt, onmt.Loss.BleuScore())
        self.model.eval()
        batch.cuda(self.gpus[0])
        for b in batch.xsplit(6):  # mini_size = 6
            tuples = sampler.sample(100, b.src, b.tgt, b.lengths,
                                    largest_len=b.lengths.data.max(),
                                    is_eval=True)
            seqs, _, probs, bleus = tuples
            probs.detach_()
            first_moment = (probs * bleus).data.sum()
            second_moment = (probs * bleus.pow(2)).data.sum()
            n_words = 20 * b.tgt.ne(onmt.Constants.PAD).data.sum()
            rl_statistics.update(RLStatistics(first_moment,
                                              second_moment,
                                              n_words,
                                              0,
                                              6))
        self.model.train()
        return rl_statistics
