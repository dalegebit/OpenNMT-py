from __future__ import division

import os

import onmt
import onmt.Markdown
import onmt.Models
import onmt.modules
import argparse
import torch
import torch.nn as nn
from torch import cuda

import math

parser = argparse.ArgumentParser(description='train.py')
onmt.Markdown.add_md_help_argument(parser)

# Data options

parser.add_argument('-data', required=True,
                    help='Path to the *-train.pt file from preprocess.py')
parser.add_argument('-save_model', default='model',
                    help="""Model filename (the model will be saved as
                    <save_model>_epochN_PPL.pt where PPL is the
                    validation perplexity""")
parser.add_argument('-train_from_state_dict', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model's state_dict.""")
parser.add_argument('-train_from', default='', type=str,
                    help="""If training from a checkpoint then this is the
                    path to the pretrained model.""")

# Model options

parser.add_argument('-layers', type=int, default=2,
                    help='Number of layers in the LSTM encoder/decoder')
parser.add_argument('-rnn_size', type=int, default=500,
                    help='Size of LSTM hidden states')
parser.add_argument('-word_vec_size', type=int, default=500,
                    help='Word embedding sizes')
parser.add_argument('-feat_vec_size', type=int, default=20,
                    help='Feature vec sizes')
parser.add_argument('-feat_merge', type=str, default='concat',
                    choices=['concat', 'sum'],
                    help='Merge action for the features embeddings')
parser.add_argument('-feat_vec_exponent', type=float, default=0.7,
                    help="""When features embedding sizes are not set and
                    using -feat_merge concat, their dimension will be set
                    to N^feat_vec_exponent where N is the number of values
                    the feature takes""")
parser.add_argument('-input_feed', type=int, default=1,
                    help="""Feed the context vector at each time step as
                    additional input (via concatenation with the word
                    embeddings) to the decoder. """)
parser.add_argument('-rnn_type', type=str, default='LSTM',
                    choices=['LSTM', 'GRU'],
                    help="""The gate type to use in the RNNs""")
# parser.add_argument('-residual',   action="store_true",
#                     help="Add residual connections between RNN layers.")
parser.add_argument('-brnn', action='store_true',
                    help='Use a bidirectional encoder')
parser.add_argument('-brnn_merge', default='concat',
                    help="""Merge action for the bidirectional hidden states:
                    [concat|sum]""")
parser.add_argument('-copy_attn', action="store_true",
                    help='Train copy attention layer.')
parser.add_argument('-coverage_attn', action="store_true",
                    help='Train a coverage attention layer.')
parser.add_argument('-lambda_coverage', type=float, default=1,
                    help='Lambda value for coverage.')

parser.add_argument('-encoder_layer', type=str, default='rnn',
                    help="""Type of encoder layer to use.
                    Options: [rnn|mean|transformer]""")
parser.add_argument('-decoder_layer', type=str, default='rnn',
                    help='Type of decoder layer to use. [rnn|transformer]')
parser.add_argument('-context_gate', type=str, default=None,
                    choices=['source', 'target', 'both'],
                    help="""Type of context gate to use [source|target|both].
                    Do not select for no context gate.""")
parser.add_argument('-attention_type', type=str, default='general',
                    choices=['dot', 'general', 'mlp'],
                    help="""The attention type to use:
                    dotprot or general (Luong) or MLP (Bahdanau)""")
parser.add_argument('-attention_use_emb', action="store_true",
                    help="""Whether add source embeddings to contexts
                    when averaging the query values""")
parser.add_argument('-multi_attn', action="store_true",
                    help="""Whether use multi-hop attention in decoder""")
parser.add_argument('-decoder_ngram', type=int, help="""Whether use n-gram histories
                    in decoder rnn""")

# Reinforce Learning options
parser.add_argument('-rl_training', action='store_true',
                    help="""Whether start reinforcement training""")
parser.add_argument('-rl_sample_num', type=int, default=5,
                    help="""The number of generated samples for each
                    source sequence""")
parser.add_argument('-max_samples_per_optim', type=int, default=100,
                    help="""The maximum number of generated samples that
                    are going to calculate gradient at a time, controlling
                    the memory usage""")
parser.add_argument('-rollout', type=int, default=20,
                    help="Refs: SeqGAN")

# Optimization options
parser.add_argument('-encoder_type', default='text',
                    help="Type of encoder to use. Options are [text|img].")
parser.add_argument('-batch_size', type=int, default=64,
                    help='Maximum batch size')
parser.add_argument('-max_generator_batches', type=int, default=32,
                    help="""Maximum batches of words in a sequence to run
                    the generator on in parallel. Higher is faster, but uses
                    more memory.""")
parser.add_argument('-epochs', type=int, default=13,
                    help='Number of training epochs')
parser.add_argument('-start_epoch', type=int, default=1,
                    help='The epoch from which to start')
parser.add_argument('-param_init', type=float, default=0.1,
                    help="""The hyper-parameters of the distribution that is used to
                    initialize the model parameters. (lower=-param_init,
                    upper=param_init) for uniform distribution, (mean=0,
                    std=param_init) for normal distribution.
                    Use 0 to not use initialization""")
parser.add_argument('-init_method', type=str, default='uniform',
                    choices=['uniform', 'normal', 'xavier', 'ortho'],
                    help="""The parameter initialization method to use.
                    Options: [uniform|normal|xavier|ortho]""")
parser.add_argument('-optim', default='sgd',
                    help="Optimization method. [sgd|adagrad|adadelta|adam]")
parser.add_argument('-max_grad_norm', type=float, default=5,
                    help="""If the norm of the gradient vector exceeds this,
                    renormalize it to have the norm equal to max_grad_norm""")
parser.add_argument('-dropout', type=float, default=0.3,
                    help='Dropout probability; applied between LSTM stacks.')
parser.add_argument('-position_encoding', action='store_true',
                    help='Use a sinusoid to mark relative words positions.')
parser.add_argument('-share_decoder_embeddings', action='store_true',
                    help='Share the word and softmax embeddings for decoder.')
parser.add_argument('-momentum', type=float, default=0.9,
                    help='Momentum')
parser.add_argument('-adam_beta1', type=float, default=0.9,
                    help='Adam beta1')
parser.add_argument('-adam_beta2', type=float, default=0.98,
                    help='Adam beta2')


parser.add_argument('-curriculum_epoches', type=int, default=0,
                    help="""For this many epochs, order the minibatches based
                    on source sequence length. Sometimes setting this to 1 will
                    increase convergence speed.""")
parser.add_argument('-extra_shuffle', action="store_true",
                    help="""By default only shuffle mini-batch order; when true,
                    shuffle and re-assign mini-batches""")
parser.add_argument('-truncated_decoder', type=int, default=0,
                    help="""Truncated bptt.""")

# learning rate
parser.add_argument('-learning_rate', type=float, default=1.0,
                    help="""Starting learning rate. If adagrad/adadelta/adam is
                    used, then this is the global learning rate. Recommended
                    settings: sgd = 1, adagrad = 0.1,
                    adadelta = 1, adam = 0.001""")
parser.add_argument('-learning_rate_decay', type=float, default=0.5,
                    help="""If update_learning_rate, decay learning rate by
                    this much if (i) perplexity does not decrease on the
                    validation set or (ii) epoch has gone past
                    start_decay_at""")
parser.add_argument('-start_decay_at', type=int, default=8,
                    help="""Start decaying every epoch after and including this
                    epoch""")
parser.add_argument('-start_checkpoint_at', type=int, default=0,
                    help="""Start checkpointing every epoch after and including this
                    epoch""")
parser.add_argument('-decay_method', type=str, default="",
                    help="""Use a custom learning rate decay [|noam] """)
parser.add_argument('-warmup_steps', type=int, default=4000,
                    help="""Number of warmup steps for custom decay.""")
parser.add_argument('-lr_schedule_interval', type=int, default=200,
                    help="""Compute validation statitisc and schedule
                    learning rate at this interval""")


# pretrained word vectors

parser.add_argument('-pre_word_vecs_enc',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the encoder side.
                    See README for specific formatting instructions.""")
parser.add_argument('-pre_word_vecs_dec',
                    help="""If a valid path is specified, then this will load
                    pretrained word embeddings on the decoder side.
                    See README for specific formatting instructions.""")

# GPU
parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help="Use CUDA on the listed devices.")

parser.add_argument('-log_interval', type=int, default=50,
                    help="Print stats at this interval.")
parser.add_argument('-log_file', type=str,
                    help="Print log information into log file.")
parser.add_argument('-log_server', type=str, default="",
                    help="Send logs to this crayon server.")
parser.add_argument('-experiment_name', type=str, default="",
                    help="Name of the experiment for logging.")

parser.add_argument('-seed', type=int, default=-1,
                    help="""Random seed used for the experiments
                    reproducibility.""")

opt = parser.parse_args()


if opt.seed > 0:
    torch.manual_seed(opt.seed)

if torch.cuda.is_available() and not opt.gpus:
    print("WARNING: You have a CUDA device, should run with -gpus 0")

if opt.gpus:
    cuda.set_device(opt.gpus[0])
    if opt.seed > 0:
        torch.cuda.manual_seed(opt.seed)


def initParam(param, method, factor):
    if method == 'uniform':
        param.data.uniform_(-factor, factor)
    elif method == 'normal':
        param.data.normal_(0, factor)
    elif method == 'xavier':
        var = 2 / (param.size(0)+param.size(0)) if param.dim() == 2 \
              else 1 / param.size(0)
        param.data.normal_(0, math.sqrt(var))
    elif method == 'ortho':
        param.data.normal_(0, factor)
        if param.dim() == 2:
            stride = math.gcd(param.size(0), param.size(1))
            for i in range(param.size(0)//stride):
                for j in range(param.size(1)//stride):
                    u, _, _ = torch.svd(param.data[i*stride:(i+1)*stride,
                                                   j*stride:(j+1)*stride])
                    param.data[i*stride:(i+1)*stride,
                               j*stride:(j+1)*stride] = u


def trainModel(trainer):

    for epoch in range(opt.start_epoch, opt.epochs + 1):
        print('')

        #  (1) train for one epoch on the training set
        train_stats = trainer.train(epoch)
        print('Train perplexity: %g' % train_stats.ppl())
        print('Train accuracy: %g' % train_stats.accuracy())
        print('Train bleu: %g' % train_stats.bleu())

        #  (2) evaluate on the validation set
        valid_stats = trainer.validate()
        print('Validation perplexity: %g' % valid_stats.ppl())
        print('Validation accuracy: %g' % valid_stats.accuracy())
        print('Validation bleu: %g' % valid_stats.bleu())

        # Log to remote server.
        if opt.log_server:
            train_stats.log("train", trainer.experiment, trainer.optim)
            valid_stats.log("valid", trainer.experiment, trainer.optim)

        #  (3) update the learning rate
        if epoch >= opt.start_checkpoint_at:
            trainer.drop_checkpoint(epoch, valid_stats)


def main():
    print("Loading data from '%s'" % opt.data)

    dataset = torch.load(opt.data)
    dict_checkpoint = (opt.train_from if opt.train_from
                       else opt.train_from_state_dict)
    if dict_checkpoint:
        print('Loading dicts from checkpoint at %s' % dict_checkpoint)
        checkpoint = torch.load(dict_checkpoint)
        dataset['dicts'] = checkpoint['dicts']

    trainData = onmt.Dataset(dataset['train']['src'],
                             dataset['train']['tgt'], opt.batch_size, opt.gpus,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['train'].get('src_features'),
                             tgtFeatures=dataset['train'].get('tgt_features'),
                             alignment=dataset['train'].get('alignments'))
    validData = onmt.Dataset(dataset['valid']['src'],
                             dataset['valid']['tgt'], opt.batch_size, opt.gpus,
                             volatile=True,
                             data_type=dataset.get("type", "text"),
                             srcFeatures=dataset['valid'].get('src_features'),
                             tgtFeatures=dataset['valid'].get('tgt_features'),
                             alignment=dataset['valid'].get('alignments'))

    dicts = dataset['dicts']
    print(' * vocabulary size. source = %d; target = %d' %
          (dicts['src'].size(), dicts['tgt'].size()))
    if 'src_features' in dicts:
        for j in range(len(dicts['src_features'])):
            print(' * src feature %d size = %d' %
                  (j, dicts['src_features'][j].size()))

    dicts = dataset['dicts']
    print(' * number of training sentences. %d' %
          len(dataset['train']['src']))
    print(' * maximum batch size. %d' % opt.batch_size)

    print('Building model...')

    model, generator = onmt.Models.build_model(opt, dicts)

    if opt.train_from:
        print('Loading model from checkpoint at %s' % opt.train_from)
        chk_model = checkpoint['model']
        generator_state_dict = chk_model.generator.state_dict()
        model_state_dict = {k: v for k, v in chk_model.state_dict().items()
                            if 'generator' not in k}
        model.load_state_dict(model_state_dict)
        generator.load_state_dict(generator_state_dict)
        opt.start_epoch = checkpoint['epoch'] + 1
    elif opt.train_from_state_dict:
        print('Loading model from checkpoint at %s'
              % opt.train_from_state_dict)
        model.load_state_dict(checkpoint['model'])
        generator.load_state_dict(checkpoint['generator'])
        opt.start_epoch = checkpoint['epoch'] + 1
    else:
        if opt.param_init != 0.0:
            print('Intializing params')
            for p in model.parameters():
                initParam(p, opt.init_method, opt.param_init)
                # p.data.uniform_(-opt.param_init, opt.param_init)
                # if len(p.size()) == 1:
                #     p.data.fill_(1.0)
                # else:
                #     p.data.uniform_(-opt.param_init, opt.param_init)

    if len(opt.gpus) >= 1:
        model.cuda()
        generator.cuda()
    else:
        model.cpu()
        generator.cpu()

    # if len(opt.gpus) > 1:
    #     print('Multi gpu training ', opt.gpus)
    #     model = nn.DataParallel(model, device_ids=opt.gpus, dim=1)
    #     generator = nn.DataParallel(generator, device_ids=opt.gpus, dim=0)

    model.generator = generator

    optim = onmt.Optim(
        opt.optim, opt.learning_rate, opt.max_grad_norm,
        lr_decay=opt.learning_rate_decay,
        start_decay_at=opt.start_decay_at,
        momentum=opt.momentum,
        beta1=opt.adam_beta1, beta2=opt.adam_beta2,
        use_noam=(opt.decay_method == 'noam')
    )

    model.encoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_enc)
    model.decoder.embeddings.load_pretrained_vectors(opt.pre_word_vecs_dec)

    optim_changed = False
    if opt.train_from_state_dict or opt.train_from:
        print('Loading optimizer from checkpoint:')
        optim._step = checkpoint['optim']['step']
        optim.method = checkpoint['optim']['method']
        if opt.learning_rate != parser.get_default('learning_rate'):
            optim.lr = opt.learning_rate
        if opt.start_decay_at != parser.get_default('start_decay_at'):
            optim.start_decay_at = opt.start_decay_at
        if opt.optim != optim.method:
            print("Change optim method %s -> %s" % (optim.method, opt.optim))
            optim.method = opt.optim
            optim_changed = True
        print(optim)

    optim.set_parameters(model.parameters())

    if opt.train_from or opt.train_from_state_dict and not optim_changed:
        optim.optimizer.load_state_dict(checkpoint['optim']['optim_state'])

    model_dirname = os.path.dirname(opt.save_model)
    if not os.path.exists(model_dirname):
        os.mkdir(model_dirname)
    assert os.path.isdir(model_dirname), "%s not a directory" % opt.save_model

    # Set up the Crayon logging server.
    experiment = None
    if opt.log_server != "":
        from pycrayon import CrayonClient
        cc = CrayonClient(hostname=opt.log_server)

        experiments = cc.get_experiment_names()
        print(experiments)
        if opt.experiment_name in experiments:
            cc.remove_experiment(opt.experiment_name)
        experiment = cc.create_experiment(opt.experiment_name)

    if opt.log_file:
        log_dirname = os.path.dirname(opt.log_file)
        if not os.path.exists(log_dirname):
            os.mkdir(log_dirname)
        assert os.path.isdir(log_dirname), "%s not a directory" % opt.log_file

    nParams = sum([p.nelement() for p in model.parameters()])
    print('* number of parameters: %d' % nParams)
    enc = 0
    dec = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            enc += param.nelement()
        elif 'decoder' in name:
            dec += param.nelement()
        else:
            print(name, param.nelement())
    print('encoder: ', enc)
    print('decoder: ', dec)

    trainer = onmt.Trainer(model, trainData, validData, dataset,
                           optim, opt, experiment)

    trainModel(trainer)


if __name__ == "__main__":
    print(opt)
    main()
