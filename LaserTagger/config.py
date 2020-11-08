import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='PyTorch LaserTagger')
    # data arguments
    parser.add_argument('--data', default='dataset_50_Laser/',
                        help='Path to dataset')
    parser.add_argument('--save', default='LaserTagger/checkpoints/',
                        help='Directory to save checkpoints in')
    parser.add_argument('--expname', type=str, default='test_50',
                        help='Name to identify experiment')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='How many workers to use in data loading')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Load checkpoint from disk')
    parser.add_argument('--pin_memory', type=bool, default=False,
                        help='pin memory for data loader')
    # model arguments
    parser.add_argument('--context_mode', type=str, default='changes',
                        help='Context mode, either "full", "none", "before", "after", "changes", "path"')
    parser.add_argument('--backbone', type=str, default='transformer',
                        help='Either lstm or transformer')
    parser.add_argument('--input_dim', default=64, type=int,
                        help='Size of input word vector')
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help='Size of hidden cell state')
    parser.add_argument('--num_of_layers', default=2, type=int,
                        help='LSTMs number of layers')
    parser.add_argument('--dropout', default=0.25, type=float,
                        help='drop probability')

    # training arguments
    # parser.add_argument('--epochs', default=15, type=int,
    #                     help='Number of total epochs to run')
    parser.add_argument('--accum_grad', default=1, type=int,
                        help='Accumulate gradient')
    parser.add_argument('--early_stopping', default=10, type=int,
                        help='Number of epochs to run after last improvement')
    parser.add_argument('--batch_size', default=32, type=int,
                        help='Batch size for optimizer updates')
    parser.add_argument('--lr', default=1e-4, type=float,
                        metavar='LR', help='Initial learning rate')
    parser.add_argument('--wd', default=0, type=float,  # default=1e-4
                        help='Weight decay')
    parser.add_argument('--sparse', action='store_true',
                        help='Enable sparsity for embeddings, \
                              incompatible with weight decay')
    parser.add_argument('--optim', default='adam',
                        help='Optimizer (default: adam)')
    parser.add_argument('--momentum', default=0, type=float,
                        help='Momentum for SGD optimizer only')

    # miscellaneous options
    parser.add_argument('--seed', default=123, type=int,
                        help='Random seed (default: 123)')
    parser.add_argument('--disable_prog_bar', action='store_true',
                        help='Disable training and testing progress bar')
    cuda_parser = parser.add_mutually_exclusive_group(required=False)
    cuda_parser.add_argument('--cuda', dest='cuda', action='store_true')
    cuda_parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    parser.add_argument('--debug', action='store_true',
                        help='Run on toy configuration')
    parser.add_argument('--inference', type=bool, default=False,
                        help='Run inference only')
    parser.set_defaults(cuda=True)

    args = parser.parse_args()
    return args
