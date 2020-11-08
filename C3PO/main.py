from __future__ import division
from __future__ import print_function

import pickle
import random
import logging
import torch
import torch.nn as nn
import torch.optim as optim
torch.backends.cudnn.enabled = False
import torch.utils.data as data
from Models.Encoder import Encoder
from Models.EncoderTxtCtx import EncoderTxtCtx
from Models.PointerDecoder import PointerDecoder
from dataset import Dataset, collate_fn
from utils import *
from trainer import Trainer
from config import parse_args

try:
    from apex import amp
    APEX_AVAILABLE = True
except ModuleNotFoundError:
    APEX_AVAILABLE = False


def main(args):
    global APEX_AVAILABLE
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

    if args.debug:
        args.input_dim = 8
        args.hidden_dim = 16
        args.batch_size = 5
        args.num_workers = 0
        args.lr = 0.05
        args.num_of_layers = 1
        args.dropout = 0
        args.early_stopping = 10
        args.max_seq_len = 13
        # args.disable_prog_bar = True
        args.shuffle_path = False
        args.data = "data_50_new/"
        logger.debug("Running on toy configuration")

    # argument validation
    args.cuda = args.cuda and torch.cuda.is_available()
    if args.cuda is True:
        logger.debug("CUDA is available")
    if APEX_AVAILABLE is True and args.cuda is True:
        logger.debug("APEX is available")
    else:
        APEX_AVAILABLE = False
    device = torch.device("cuda:0" if args.cuda else "cpu")
    logger.debug(args)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True  # cause bugs!!! in Tensor.topk with NaN
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    path_vocab = torch.load(os.path.join(args.data, 'path.pth'))
    logger.debug('==> path vocabulary size:         {} '.format(path_vocab.size()))
    src_tgt_vocab = torch.load(os.path.join(args.data, 'src_tgt.pth'))
    logger.debug('==> src_tgt vocabulary size:      {} '.format(src_tgt_vocab.size()))
    position_vocab = torch.load(os.path.join(args.data, 'position.pth'))
    logger.debug('==> position vocabulary size:     {} '.format(position_vocab.size()))
    if args.context_mode == 'txt':
        ctx_vocab = torch.load(os.path.join(args.data, 'ctx.pth'))
        logger.debug('==> context vocabulary size:      {} '.format(ctx_vocab.size()))


    train_dir = os.path.join(args.data, 'train/')
    dev_dir = os.path.join(args.data, 'dev/')
    test_dir = os.path.join(args.data, 'test/')

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

    train_set = Dataset(train_dir, device, args)
    train_generator = data.DataLoader(train_set, **train_params)
    logger.debug('==> Size of train data:           {} '.format(len(train_set)))

    dev_set = Dataset(dev_dir, device, args)
    dev_generator = data.DataLoader(dev_set, **test_params)
    logger.debug('==> Size of dev data:             {} '.format(len(dev_set)))

    test_set = Dataset(test_dir, device, args)
    test_generator = data.DataLoader(test_set, **test_params)
    logger.debug('==> Size of test data:            {} '.format(len(test_set)))

    if args.context_mode == 'txt':
        encoder = EncoderTxtCtx(
            path_vocab_size=path_vocab.size(),
            src_tgt_vocab_size=src_tgt_vocab.size(),
            position_vocab_size=position_vocab.size(),
            ctx_vocab_size=ctx_vocab.size(),
            in_dim=args.input_dim,
            h_dim=args.hidden_dim,
            num_layers=args.num_of_layers,
            dropout=args.dropout,
            device=device,
        )
    else:
        encoder = Encoder(
            path_vocab_size=path_vocab.size(),
            src_tgt_vocab_size=src_tgt_vocab.size(),
            position_vocab_size=position_vocab.size(),
            in_dim=args.input_dim,
            h_dim=args.hidden_dim,
            num_layers=args.num_of_layers,
            dropout=args.dropout,
            device=device,
            ctx_mode=args.context_mode
        )

    decoder = PointerDecoder(
        in_dim=args.hidden_dim,
        h_dim=args.hidden_dim,
        num_of_layers=args.num_of_layers,
        device=device,
        dropout=args.dropout,
        use_attention=args.attention
    )

    logger.debug('==> Total trainable parameters:   {} '.format((count_parameters(encoder) + count_parameters(decoder))))
    encoder.to(device)
    decoder.to(device)
    parameters = filter(lambda p: p.requires_grad, list(encoder.parameters()) + list(decoder.parameters()))
    if args.optim == 'adam':
        optimizer = optim.Adam(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'adagrad':
        optimizer = optim.Adagrad(parameters, lr=args.lr, weight_decay=args.wd)
    elif args.optim == 'sgd':
        optimizer = optim.SGD(parameters, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)
    criterion = nn.CrossEntropyLoss(ignore_index=Constants.PAD)
    criterion.to(device)

    if APEX_AVAILABLE:
        [encoder, decoder], optimizer = amp.initialize(
            [encoder, decoder], optimizer, opt_level="O2",
            keep_batchnorm_fp32=True, loss_scale="dynamic"
        )

    # create trainer object for training and testing
    trainer = Trainer(args, encoder, decoder, optimizer, criterion, device, args.disable_prog_bar)

    best = -float('inf')
    counter = 0
    epoch = 0

    if args.load_checkpoint != '' and os.path.exists(args.load_checkpoint):
        logger.debug('==> Loading checkpoint "{}" from disk'.format(args.load_checkpoint))
        checkpoint = torch.load(args.load_checkpoint, map_location=torch.device(device))
        trainer.encoder.load_state_dict(checkpoint['encoder'])
        trainer.decoder.load_state_dict(checkpoint['decoder'])
        if APEX_AVAILABLE:
            amp.load_state_dict(checkpoint['amp'])
        best = checkpoint['dev_acc']
        epoch = checkpoint['epoch'] + 1
        trainer.epoch = checkpoint['epoch'] + 1
        if args.inference:
            logger.debug('==> Running inference on test set')
            test_loss, res_test = trainer.test(test_generator)
            test_precision, test_recall, test_f1, test_acc = res_test['precision'], res_test['recall'], res_test['f1'], \
                                                         res_test['acc']
            logger.info(
                '==> Epoch {}, Test\t\tLoss: {:0.3f}\tAcc: {:0.3f}'.format(
                    epoch + 1, test_loss, test_acc))
            mapping = dict()
            for i, (project_name, idx) in enumerate(res_test['ids']):
                if project_name not in mapping:
                    mapping[project_name] = list()
                mapping[project_name].append((idx, res_test['predicted_ops'][i]))
            for k in mapping.keys():
                _, mapping[k] = zip(*sorted(mapping[k], key=lambda t: t[0]))
            with open(os.path.join(args.save, args.expname + "_test_results.pickle"), "wb") as f:
                pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)
            return

    while True:
        train_loss, res_train = trainer.train(train_generator)
        train_precision, train_recall, train_f1, train_acc = res_train['precision'], res_train['recall'], res_train['f1'], res_train['acc']
        logger.info('==> Epoch {}, Train\t\tLoss: {:0.3f}\tAcc: {:0.3f}'.format(
            epoch + 1, train_loss, train_acc))
        dev_loss, res_dev = trainer.test(dev_generator)
        dev_precision, dev_recall, dev_f1, dev_acc = res_dev['precision'], res_dev['recall'], res_dev['f1'], res_dev['acc']
        logger.info('==> Epoch {}, Dev\t\tLoss: {:0.3f}\tAcc: {:0.3f}'.format(
            epoch + 1, dev_loss, dev_acc))
        if best < dev_acc:
            best = dev_acc
            checkpoint = {
                'encoder': trainer.encoder.state_dict(),
                'decoder': trainer.decoder.state_dict(),
                'optim': trainer.optimizer,
                'dev_acc': dev_acc,
                'args': args, 'epoch': epoch,
                'amp': amp.state_dict() if APEX_AVAILABLE else None
            }
            logger.debug('==> New optimum found, checkpointing everything now...')
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            mapping = dict()
            for i, (project_name, idx) in enumerate(res_dev['ids']):
                if project_name not in mapping:
                    mapping[project_name] = list()
                mapping[project_name].append((idx, res_dev['predicted_ops'][i]))
            for k in mapping.keys():
                _, mapping[k] = zip(*sorted(mapping[k], key=lambda t: t[0]))
            with open(os.path.join(args.save, args.expname + "_dev_results.pickle"), "wb") as f:
                pickle.dump(mapping, f, protocol=pickle.HIGHEST_PROTOCOL)

            counter = 0
        else:
            counter += 1
        logger.debug('')
        if counter >= args.early_stopping:
            logger.debug('==> {} epochs have been passed without Acc improvement, running inference on test:'.format(counter))
            checkpoint = torch.load('%s.pt' % os.path.join(args.save, args.expname))
            trainer.encoder.load_state_dict(checkpoint['encoder'])
            trainer.transformer.load_state_dict(checkpoint['transformer'])
            trainer.operation_mix.load_state_dict(checkpoint['operation_mix'])
            trainer.decoder.load_state_dict(checkpoint['decoder'])
            test_loss, test_acc, test_precision, test_recall, test_f1 = trainer.test(test_generator)
            checkpoint['test_accuracy'] = test_acc
            torch.save(checkpoint, '%s.pt' % os.path.join(args.save, args.expname))
            logger.info('==> Epoch {}, Test\t\tLoss: {}\tAcc: {}'.format(
                checkpoint['epoch'] + 1, test_loss, test_acc))
        
            return
        epoch += 1


if __name__ == "__main__":
    global args
    args = parse_args()
    main(args)

