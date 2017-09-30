import torch
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


def train_agent(rank, opt, dicts, queue, out_queue, model, locks, shard_size):
    local_model, local_gen = onmt.Models.build_model(opt, dicts)
    local_model.generator = local_gen
    local_model.cuda(rank)
    sampler = onmt.Sampler(local_model, opt, onmt.Loss.BleuScore())
    local_statistics = RLStatistics()
    while True:
        batch = queue.get()
        batch.cuda(rank)
        local_model.load_state_dict(model.state_dict())
        local_model.zero_grad()
        for b in batch.xsplit(shard_size):
            samples, _, probs, scores = \
                sampler.sample(opt.rl_sample_num, b.src, b.tgt, b.lengths,
                               largest_len=b.lengths.data.max())
            total_loss = (probs * scores).sum() / opt.batch_size
            total_loss.backward()
            probs_detach = probs.detach()
            first_moment = (probs_detach * scores).data.sum()
            second_moment = (probs_detach * scores.pow(2)).data.sum()
            n_words = opt.rl_sample_num * \
                b.tgt.ne(onmt.Constants.PAD).data.sum()
            local_statistics.update(
                RLStatistics(first_moment, second_moment,
                             n_words, shard_size))
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
        sampler = onmt.Sampler(self.model, self.opt, onmt.Loss.BleuScore())
        self.model.eval()
        batch.cuda(self.gpus[0])
        for b in batch.xsplit(self.shard_size):  # mini_size = 6
            tuples = sampler.sample(self.sample_num, b.src, b.tgt, b.lengths,
                                    largest_len=b.lengths.data.max(),
                                    is_eval=True)
            seqs, _, probs, bleus = tuples
            probs.detach_()
            first_moment = (probs * bleus).data.sum()
            second_moment = (probs * bleus.pow(2)).data.sum()
            n_words = self.sample_num * b.tgt.ne(onmt.Constants.PAD).data.sum()
            statistics.update(
                RLStatistics(first_moment, second_moment, n_words,
                             self.shard_size))
        self.model.train()
        return statistics


class Trainer(object):
    def __init__(self, model, train_data, valid_data, dataset,
                 optim, opt, experiment_name=''):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.optim = optim
        self.opt = opt
        self.experiment = experiment_name
        self.model.train()

        # Define criterion of each GPU.
        if not opt.copy_attn:
            self.criterion = onmt.Loss.NMTCriterion(
                dataset['dicts']['tgt'].size(),
                opt)
        else:
            self.criterion = onmt.modules.CopyCriterion

        if opt.rl_training:
            self.rl_trainer = RLBatchTrainer(self.model, dataset['dicts'], opt)

    def train(self, epoch):
        if self.opt.extra_shuffle and epoch > self.opt.curriculum_epoches:
            self.train_data.shuffle()

        mem_loss = onmt.Loss.MemoryEfficientLoss(self.opt, self.model.generator,
                                                 self.criterion,
                                                 copy_loss=self.opt.copy_attn)

        # Shuffle mini batch order.
        batch_order = torch.randperm(len(self.train_data))

        if self.opt.rl_training:
            total_stats = onmt.Loss.RLStatistics()
            report_stats = onmt.Loss.RLStatistics()
        else:
            total_stats = onmt.Loss.Statistics()
            report_stats = onmt.Loss.Statistics()

        for i in range(len(self.train_data)):
            batchIdx = batch_order[i] \
                if epoch > self.opt.curriculum_epoches else i
            batch = self.train_data[batchIdx]
            target_size = batch.tgt.size(0)

            if self.opt.rl_training:
                self.model.zero_grad()
                rl_stats = self.rl_trainer.get_batch_grad(batch)
                self.optim.step()
                total_stats.update(rl_stats)
                report_stats.update(rl_stats)
                report_stats.n_src_words += self.opt.rl_sample_num + \
                    batch.lengths.data.sum()
            else:
                dec_state = None
                trunc_size = self.opt.truncated_decoder \
                    if self.opt.truncated_decoder else target_size
                if len(self.opt.gpus):
                    batch.cuda(self.opt.gpus[0])
                for j in range(0, target_size-1, trunc_size):
                    trunc_batch = batch.truncate(j, j + trunc_size)

                    # Main training loop
                    self.model.zero_grad()
                    outputs, attn, dec_state = \
                        self.model(trunc_batch.src,
                                   trunc_batch.tgt,
                                   trunc_batch.lengths,
                                   dec_state,
                                   largest_len=trunc_batch.lengths.data.max())

                    batch_stats, inputs, grads \
                        = mem_loss.loss(trunc_batch, outputs, attn)

                    torch.autograd.backward(inputs, grads)

                    # Update the parameters.
                    self.optim.step()
                    total_stats.update(batch_stats)
                    report_stats.update(batch_stats)
                    if dec_state is not None:
                        dec_state.detach()

                report_stats.n_src_words += batch.lengths.data.sum()

            if i % self.opt.log_interval == -1 % self.opt.log_interval:
                report_stats.output(epoch, i+1, len(self.train_data),
                                    total_stats.start_time,
                                    log_file=self.opt.log_file)

                if self.opt.log_server:
                    report_stats.log("progress", self.experiment, self.optim)
                if self.opt.rl_training():
                    report_stats = onmt.Loss.RLStatistics()
                else:
                    report_stats = onmt.Loss.Statistics()

        return total_stats
