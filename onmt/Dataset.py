from __future__ import division

import math
import random

import torch
from torch.autograd import Variable

import onmt


class Dataset(object):

    def __init__(self, srcData, tgtData, batchSize, cuda, volatile=False, sortBuffer=True, bufferRatio=10):
        self.src = srcData
        if tgtData:
            self.tgt = tgtData
            assert(len(self.src) == len(self.tgt))
        else:
            self.tgt = None
        self.cuda = cuda

        self.batchSize = batchSize
        self.bufferSize = batchSize * bufferRatio
        self.numBatches = int(math.ceil(len(self.src)/float(batchSize)))
        self.volatile = volatile
        self.sortBuffer = sortBuffer

    def _batchify(self, data, align_right=False, include_lengths=False):
        lengths = [x.size(0) for x in data]
        max_length = max(lengths)
        out = data[0].new(len(data), max_length).fill_(onmt.Constants.PAD)
        for i in range(len(data)):
            data_length = data[i].size(0)
            offset = max_length - data_length if align_right else 0
            out[i].narrow(0, offset, data_length).copy_(data[i])

        if include_lengths:
            return out, lengths
        else:
            return out


    def __getitem__(self, index):
        assert index < self.numBatches, "%d > %d" % (index, self.numBatches)
        srcBatch, lengths = self._batchify(
            self.src[index*self.batchSize:min((index+1)*self.batchSize, len(self.src))],
            align_right=False, include_lengths=True)

        if self.tgt:
            tgtBatch = self._batchify(
                self.tgt[index*self.batchSize:min((index+1)*self.batchSize, len(self.tgt))])
        else:
            tgtBatch = None

        # within batch sorting by decreasing length for variable length rnns
        indices = range(len(srcBatch))
        batch = zip(indices, srcBatch) if tgtBatch is None else zip(indices, srcBatch, tgtBatch)
        batch, lengths = zip(*sorted(zip(batch, lengths), key=lambda x: -x[1]))
        if tgtBatch is None:
            indices, srcBatch = zip(*batch)
        else:
            indices, srcBatch, tgtBatch = zip(*batch)

        def wrap(b):
            if b is None:
                return b
            b = torch.stack(b, 0).t().contiguous()
            if self.cuda:
                b = b.cuda()
            b = Variable(b, volatile=self.volatile)
            return b

        # wrap lengths in a Variable to properly split it in DataParallel
        lengths = torch.LongTensor(lengths).view(1, -1)
        lengths = Variable(lengths, volatile=self.volatile)

        return (wrap(srcBatch), lengths), wrap(tgtBatch), indices

    def __len__(self):
        return self.numBatches


    def shuffle(self):
        data = list(zip(self.src, self.tgt))
        data = [data[i] for i in torch.randperm(len(data))]
        if self.sortBuffer:
           numBuffers = int(math.ceil(len(data)/float(self.bufferSize)))
           for i in range(numBuffers):
               higherb = min(i*self.bufferSize, len(data))
               data[i*self.bufferSize:higherb] = sorted(data[i*self.bufferSize:higherb], key=lambda x: -len(x[0]))
        self.src, self.tgt = zip(*data)
