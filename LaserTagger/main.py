from __future__ import division
from __future__ import print_function

import os
import pickle
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.enabled = False
import torch.utils.data as data
from Models.SequenceTagger import SequenceTagger
from Models.BiLSTMCRF import BiLSTM_CRF
from Models.TransformerCRF_V2 import Transformer_CRF
# from Models.BiLSTMCRF_V2 import BiLSTM_CRF_V2
from dataset import Dataset, collate_fn
from utils import *
from trainer import Trainer
from config import parse_args
try:
    from notify_run import Notify
    notify = Notify()
    NOTIFY_AVAILABLE = True
except ModuleNotFoundError:
    NOTIFY_AVAILABLE = False

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False

def main():
    global args
    global APEX_AVAILABLE
    args = parse_args()

    # global logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s:%(name)s:%(message)s")
    # file logger
    if not os.path.exists(args.save):
        os.mkdir(args.save)
    dfh = logging.FileHandler(os.path.join(args.save, args.expname)+'_DEBUG.log', mode='w')
    dfh.setLevel(logging.DEBUG)
    dfh.setFormatter(formatter)
    logger.addHandler(dfh)

    fh = logging.FileHandler(os.path.join(args.save, args.expname)+'.log', mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    # console logger
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda is True:
        logger.debug("CUDA is available")
    if APEX_AVAILABLE is True and args.cuda is True:
        logger.debug("APEX is available")
    else:
        APEX_AVAILABLE = False
    device = torch.device("cuda:0" if args.cuda else "cpu")
    if args.sparse and args.wd != 0:
        logger.error('Sparsity and weight decay are incompatible, pick one!')
        exit()
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True  # cause bugs!!! in Tensor.topk with NaN
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    src_vocab = torch.load(os.path.join(args.data, 'src.pth'))
    logger.debug('==> source vocabulary size:       {} '.format(src_vocab.size()))
    tgt_vocab = torch.load(os.path.join(args.data, 'tgt.pth'))
    logger.debug('==> target vocabulary size:       {} '.format(tgt_vocab.size()))
    ctx_vocab = torch.load(os.path.join(args.data, 'ctx.pth'))
    logger.debug('==> ctx vocabulary size:          {} '.format(ctx_vocab.size()))

    train_path = os.path.join(args.data, 'train.pth')
    dev_path = os.path.join(args.data, 'dev.pth')
    test_path = os.path.join(args.data, 'test.pth')

    train_params = {'batch_size': args.batch_size,
              'shuffle': True,
              'num_workers': args.num_workers,
              'collate_fn': collate_fn,
              'pin_memory': args.pin_memory}
    test_params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': 0,
              'collate_fn': collate_fn,
              'pin_memory': args.pin_memory}

    train_set = Dataset(train_path, device, ctx_vocab, args)
    train_generator = data.DataLoader(train_set, **train_params)
    logger.debug('==> Size of train data:           {} '.format(len(train_set)))

    dev_set = Dataset(dev_path, device, ctx_vocab, args)
    dev_generator = data.DataLoader(dev_set, **test_params)
    logger.debug('==> Size of dev data:             {} '.format(len(dev_set)))

    test_set = Dataset(test_path, device, ctx_vocab, args)
    test_generator = data.DataLoader(test_set, **test_params)
    logger.debug('==> Size of test data:            {} '.format(len(test_set)))

    # initialize model, criterion/loss_function, optimizer
    if args.backbone == 'transformer':
        model = Transformer_CRF(
            vocab_size=src_vocab.size(),
            ctx_vocab_size=ctx_vocab.size(),
            nb_labels=tgt_vocab.size(),
            emb_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            bos_idx=tgt_vocab.getIndex(Constants.BOS_WORD),
            eos_idx=tgt_vocab.getIndex(Constants.EOS_WORD),
            pad_idx=tgt_vocab.getIndex(Constants.PAD_WORD),
            num_lstm_layers=args.num_of_layers,
            dropout=args.dropout,
            device=device
        )
    else:
        model = BiLSTM_CRF(
            vocab_size=src_vocab.size(),
            ctx_vocab_size=ctx_vocab.size(),
            nb_labels=tgt_vocab.size(),
            emb_dim=args.input_dim,
            hidden_dim=args.hidden_dim,
            bos_idx=tgt_vocab.getIndex(Constants.BOS_WORD),
            eos_idx=tgt_vocab.getIndex(Constants.EOS_WORD),
            pad_idx=tgt_vocab.getIndex(Constants.PAD_WORD),
            num_lstm_layers=args.num_of_layers,
            dropout=args.dropout,
            device=device
        )

    logger.debug('==> Total trainable parameters:   {} '.format((count_parameters(model))))
    model.to(device)
    parameters = filter(lambda p: p.requires_grad, list(model.parameters()))
    if args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    if APEX_AVAILABLE:
        model, optimizer = amp.initialize(
            model, optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )

    # create trainer object for training and testing
    trainer = Trainer(args, tgt_vocab, model, optimizer, device, args.disable_prog_bar)

    best = -float('inf')
    counter = 0
    epoch = 0

    if args.checkpoint != '' and os.path.exists(args.checkpoint):
        logger.debug('==> Loading checkpoint "{}" from disk'.format(args.checkpoint))
        checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
        trainer.model.load_state_dict(checkpoint['model'])
        if APEX_AVAILABLE:
            amp.load_state_dict(checkpoint['amp'])
        best = checkpoint['dev_acc']
        epoch = checkpoint['epoch'] + 1
        trainer.epoch = checkpoint['epoch'] + 1
        if args.inference:
            logger.debug('==> Running inference on test set')
            test_acc = trainer.test(test_generator)
            logger.info('==> Epoch {}, Test\t\tAcc: {:0.3f}'.format(epoch + 1, test_acc))
            return

    while True:
        train_loss = trainer.train(train_generator)
        logger.info('==> Epoch {}, Train\t\tLoss: {:0.3f}'.format(
            epoch + 1, train_loss))
        dev_acc = trainer.test(dev_generator)
        logger.info('==> Epoch {}, Dev\t\tAcc: {:0.3f}'.format(
            epoch + 1, dev_acc))
        if best < dev_acc:
            if NOTIFY_AVAILABLE:
                notify.send('LaserTagger Epoch {}, Dev\t\tAcc: {:0.3f}'.format(
            epoch + 1, dev_acc))
            best = dev_acc
            checkpoint = {
                'model': trainer.model.state_dict(),
                'optim': trainer.optimizer,
                'dev_acc': dev_acc,
                'args': args, 'epoch': epoch,
                'amp': amp.state_dict() if APEX_AVAILABLE else None
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            counter = 0
        else:
            counter += 1
        logger.debug('')
        if counter >= args.early_stopping:
            logger.debug('==> {} epochs have been passed without Acc improvement, running inference on test:'.format(counter))
            checkpoint = torch.load('%s.pt' % os.path.join(args.save, args.expname))
            trainer.model.load_state_dict(checkpoint['model'])
            test_loss, test_acc, = trainer.test(test_generator)
            checkpoint['test_acc'] = test_acc
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            logger.info('==> Epoch {}, Test\t\tLoss: {}\tAcc: {}'.format(
                checkpoint['epoch'] + 1, test_loss, test_acc))
            return
        epoch += 1


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        if NOTIFY_AVAILABLE:
            notify.send("LaserTagger failed: {}".format(e))
