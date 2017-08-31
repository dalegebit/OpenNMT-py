from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.multiprocessing as mp
import onmt
import onmt.modules
from onmt.Loss import RLStatistics
import os



class MRTAgent(nn.Module):

    def __init__(self, model, generator, scorer, opt):
        super(MRTAgent, self).__init__()
        self.model = model
        self.generator = generator
        self.scorer = scorer
        self.use_gpu = len(opt.gpus) > 0
        self.multi_gpu = len(opt.gpus) > 1
        # self.max_len = opt.max_sent_length
        self.max_len = 50
        self.max_len_ratio = 1.5
        # self.n_sample = opt.rl_sample_size
        self.n_sample = 100
        self.alpha = 5e-3

    def forward(self, src, tgt, lengths, largest_len=None):
        tt = torch.cuda if self.use_gpu else torch

        # 1. Encode
        encStates, contexts = self.model.encoder(
            src, lengths=lengths, largest_len=largest_len
        )
        seqL, batchSize, rnnSize = contexts.size()

        contexts.retain_grad()

        encStates = self.model.init_decoder_state(contexts, encStates)

        padMask = src[:, :, 0].data.eq(onmt.Constants.PAD).t()

        def mask(padMask):
            self.model.decoder.attn.applyMask(padMask)

        totalLoss = Variable(contexts.data.new([[0]]))  # -firstMoment
        secondMoment = Variable(contexts.data.new([[0]]))
        numCorrect = Variable(contexts.data.new([[0]]))
        numWords = Variable(contexts.data.new([[0]]))

        # 2. For each encoded sample in a batch, decode
        for encState, context, pad, src_t, targ_t in zip(encStates.split(1, 1, 0),
                                                         contexts.split(1, 1),
                                                         padMask.split(1, 0),
                                                         src.split(1, 1),
                                                         tgt.split(1, 1)):

            # a. Prepare anything for every sequence to sample
            encState.expandAsBatch_(self.n_sample)
            contexts_ = context.expand(seqL, self.n_sample, rnnSize)
            padMask = pad.expand(self.n_sample, seqL)

            targ_t = targ_t.expand(targ_t.size(0), self.n_sample).t()
            src_t = src_t.expand(src_t.size(0), self.n_sample, src_t.size(2))

            max_len = int(self.max_len_ratio * lengths.data.max())+2

            decStates = encState
            sequences = []
            scores = []
            # Pointing to the index of next word
            batchIdxs = [[i] for i in range(self.n_sample)]
            curBatchIdxs = batchIdxs
            input = Variable(targ_t.data.new(1, self.n_sample).fill_(onmt.Constants.BOS))
            sequences.append(input.squeeze(0).data)
            scores.append(Variable(contexts.data.new(self.n_sample).zero_()))
            n_old_active = self.n_sample

            # b. Begin Sampling
            for i in range(1, max_len):
                mask(padMask.unsqueeze(0))
                decOut, decStates, attn = self.model.decoder(input,
                                                             src_t,
                                                             contexts_,
                                                             decStates)

                decOut = decOut.squeeze(0)
                out = self.generator.forward(decOut)
                pred_t = out.exp().multinomial(1).detach()

                active = Variable(pred_t.data.ne(onmt.Constants.EOS).squeeze()
                                                                    .nonzero()
                                                                    .squeeze())
                newBatchIdxs = []
                for n, b in enumerate(list(active.data)):
                    curBatchIdxs[b].append(n)
                    newBatchIdxs.append(curBatchIdxs[b])
                curBatchIdxs = newBatchIdxs

                score_t = out.gather(1, pred_t).squeeze(-1)
                scores.append(score_t)
                sequences.append(pred_t.squeeze().data)

                if active.size(0) == 0:
                    break
                if active.size(0) == n_old_active:
                    continue
                else:
                    n_old_active = active.size(0)

                input = pred_t.t().index_select(1, active)
                src_t = src_t.index_select(1, active)
                contexts_ = contexts_.index_select(1, active)
                padMask = padMask.index_select(0, active.data)
                decStates.activeUpdate_(active)

            # c. Package everything up, and calculate the score
            resultSeqs = [[] for _ in range(self.n_sample)]
            accumScores = [Variable(contexts.data.new([0]))] * self.n_sample

            for i in range(len(sequences)):
                pred = sequences[i]
                score = scores[i]
                for b, batchIdx in enumerate(batchIdxs):
                    if i < len(batchIdx):
                        resultSeqs[b].append(pred[batchIdx[i]])
                        accumScores[b] += score[batchIdx[i]]

            accumScores = torch.cat(accumScores, 0)

            bleu = self.scorer.score(resultSeqs, targ_t)
            bleu = Variable(contexts.data.new(bleu), requires_grad=True)
            prob = nn.functional.softmax(accumScores*self.alpha)
            loss = - bleu * prob

            # import os
            # from tensorboardX import SummaryWriter
            # import random
            # import time
            # time.sleep(random.randint(1, 10))
            # if not os.path.exists('runs/graph_view'):
            #     writer = SummaryWriter(log_dir='runs/graph_view')
            #     writer.add_graph(self.model.encoder, contexts)
            #     writer.close()

            # import pdb; pdb.set_trace()

            totalLoss += loss.sum()

            # For monitoring purposes
            secondMoment += (bleu.pow(2) * prob.detach()).sum()

            numWords += targ_t.data.ne(onmt.Constants.PAD).sum()
            # print sequences.size(), targ_t.size()
            # if sequences.size(1) >= targ_t.size(1):
            #     sequences = sequences[:, :targ_t.size(1)]
            #     non_padding = targ_t.ne(onmt.Constants.PAD).data
            #     numWords += non_padding.sum()
            #     numCorrect += sequences.data.eq(targ_t.data) \
            #                            .masked_select(non_padding) \
            #                            .sum()
            # else:
            #     numWords += targ_t.ne(onmt.Constants.PAD).data.sum()
            #     targ_t = targ_t[:, :sequences.size(1)]
            #     non_padding = targ_t.ne(onmt.Constants.PAD).data
            #     numCorrect += sequences.data.eq(targ_t.data) \
            #                            .masked_select(non_padding) \
            #                            .sum()

        self.model.decoder.attn.removeMask()
        return totalLoss, secondMoment, numWords


class MRT(object):

    def __init__(self, model, generator, scorer, opt):
        self.step = MRTAgent(model, generator, scorer, opt)
        self.model = model
        self.generator = generator
        self.gpus = opt.gpus
        self.mini_size = 4

    # def a3c():
    #     os.environ['OMP_NUM_THREADS'] = '1'
    #
    #     shared_model = ActorCritic(
    #         env.observation_space.shape[0], env.action_space)
    #     shared_model.share_memory()
    #
    #     if args.no_shared:
    #         optimizer = None
    #     else:
    #         optimizer = my_optim.SharedAdam(shared_model.parameters(), lr=args.lr)
    #         optimizer.share_memory()
    #
    #     processes = []
    #
    #     p = mp.Process(target=test, args=(args.num_processes, args, shared_model))
    #     p.start()
    #     processes.append(p)
    #
    #     for rank in range(0, args.num_processes):
    #         p = mp.Process(target=train, args=(rank, args, shared_model, optimizer))
    #         p.start()
    #         processes.append(p)
    #     for p in processes:
    #         p.join()

    def policy_grad(self, batch):
        rl_statistics = RLStatistics()
        for b in batch.xsplit(self.mini_size):
            # import pdb; pdb.set_trace()
            total_loss, second_moment, n_words = \
                                    self.step(b.src, b.tgt, b.lengths,
                                              largest_len=b.lengths.data.max())

            sum_loss = total_loss.sum()
            sum_loss.div(batch.batchSize).backward()
            rl_statistics.update(RLStatistics(-sum_loss.data[0],
                                              second_moment.data.sum(),
                                              n_words.data.sum(),
                                              0,
                                              self.mini_size))
        for p in self.model.parameters():
            print p.grad.data.sum()
        print 'damn'
        for p in self.step.module.model.parameters():
            print p.grad.data.sum()
        print 'damn1'

        return rl_statistics

    # def train_agent(self, batch):
    #     rl_statistics = RLStatistics()
    #     for b in batch.xsplit(self.mini_size):
    #         # import pdb; pdb.set_trace()
    #         total_loss, second_moment, n_words = \
    #                                 self.step(b.src, b.tgt, b.lengths,
    #                                           largest_len=b.lengths.data.max())
    #
    #         sum_loss = total_loss.sum()
    #         import pdb; pdb.set_trace()
    #         sum_loss.div(batch.batchSize).backward()
    #         rl_statistics.update(RLStatistics(-sum_loss.data[0],
    #                                           second_moment.data.sum(),
    #                                           n_words.data.sum(),
    #                                           0,
    #                                           self.mini_size))
    #     for p in self.model.parameters():
    #         print p.grad.data.sum()
    #     print 'damn'
    #     for p in self.step.module.model.parameters():
    #         print p.grad.data.sum()
    #     print 'damn1'
    #
    #     return rl_statistics
