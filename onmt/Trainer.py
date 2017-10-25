import torch
import torch.nn as nn
import onmt
import onmt.modules
from onmt.RLUtil import RLBatchTrainer


class Trainer(object):
    def __init__(self, model, train_data, valid_data, dataset,
                 optim, opt, experiment=None):
        self.model = model
        self.train_data = train_data
        self.valid_data = valid_data
        self.dataset = dataset
        self.optim = optim
        self.opt = opt
        self.experiment = experiment
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

        mem_loss = onmt.Loss.MemoryEfficientLoss(self.opt,
                                                 self.model.generator,
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
                                   align=trunc_batch.lengths.data.max())

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

            if i % self.opt.lr_schedule_interval == -1 % self.opt.lr_schedule_interval:
                valid_stats = self.validate()
                self.optim.updateLearningRate(valid_stats.ppl(), epoch)

            if i % self.opt.log_interval == -1 % self.opt.log_interval:
                report_stats.output(epoch, i+1, len(self.train_data),
                                    total_stats.start_time)

                if self.opt.log_server:
                    report_stats.log("progress", self.experiment, self.optim)
                if self.opt.rl_training:
                    report_stats = onmt.Loss.RLStatistics()
                else:
                    report_stats = onmt.Loss.Statistics()

        return total_stats

    def validate(self):
        data = self.valid_data
        stats = onmt.Loss.Statistics()
        self.model.eval()
        loss = onmt.Loss.MemoryEfficientLoss(self.opt, self.model.generator,
                                             self.criterion, eval=True,
                                             copy_loss=self.opt.copy_attn,
                                             calc_bleu=True)
        for i in range(len(data)):
            batch = data[i]
            if len(self.opt.gpus):
                batch = batch.cuda(self.opt.gpus[0])
            outputs, attn, dec_hidden = \
                self.model(batch.src, batch.tgt, batch.lengths,
                           align=batch.lengths.data.max())
            batch_stats, _, _ = loss.loss(batch, outputs, attn)
            stats.update(batch_stats)
        self.model.train()
        return stats

    def drop_checkpoint(self, epoch, valid_stats):
        """ Called conditionally each epoch to save a snapshot. """
        real_model = (self.model.module
                      if isinstance(self.model, nn.DataParallel)
                      else self.model)
        real_generator = (real_model.generator.module
                          if isinstance(real_model.generator, nn.DataParallel)
                          else real_model.generator)

        model_state_dict = real_model.state_dict()
        model_state_dict = {k: v for k, v in model_state_dict.items()
                            if 'generator' not in k}
        generator_state_dict = real_generator.state_dict()
        optim_state_dict = self.optim.optimizer.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'generator': generator_state_dict,
            'dicts': self.dataset['dicts'],
            'opt': self.opt,
            'epoch': epoch,
            'optim': {'step': self.optim._step,
                      'method': self.optim.method,
                      'optim_state': optim_state_dict}
        }
        torch.save(checkpoint,
                   '%s_acc_%.2f_ppl_%.2f_e%d.pt'
                   % (self.opt.save_model, valid_stats.accuracy(),
                      valid_stats.ppl(), epoch))
