'''
@file: run.py

The script for running the experiments.

@author: Rukmangadh Sai Myana
@mail: rukman.sai@gmail.com
'''


from __future__ import print_function
import argparse
import torch
import os
from main import main
from util import get_summary_dir


def eval_str(x):
    return eval(x)


# --------------------------------- script ---------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script for '
                                     'experimentation.')
    # general flags
    parser.add_argument('--download_data', action='store_true', help='Whether '
                        'to download the data or not.')
    parser.add_argument('--data_folder', default='./data/', type=str,
                        help='The path to the folder where the data is '
                        'present.')
    parser.add_argument('--use_cuda', action='store_true', help='Whether to '
                        'use gpu for training the model or not.')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to '
                        'fetch data from pinned memory.')
    parser.add_argument('--num_epochs', default=30, type=int, help='The '
                        'number of epochs in the training')
    parser.add_argument('--train_batch_size', default=64, type=int,
                        help='The batch size used for training.')
    parser.add_argument('--test_batch_size', default=64, type=int,
                        help='The batch size used for testing.')
    parser.add_argument('--workers', default=os.cpu_count(), type=int,
                        help='The number of workers to be used for data '
                        'loading.')
    parser.add_argument('--logdir', default='./logs', type=str, help='The '
                        'directory where the logs and logging related stuff '
                        'are to be stored')

    # optimizer flags
    parser.add_argument('--optimizer', default='adam', type=str,
                        choices=['adam'], help='The optimizer to use while '
                        'training.')
    parser.add_argument('--lr', type=float, help='Learning rate for the '
                        'optimizer.')
    parser.add_argument('--betas', nargs='+', type=float, help='The *betas* '
                        'option for the optimizer. See pytorch docs.')
    parser.add_argument('--eps', type=float, help='The *eps* option for the '
                        'optimizer. See pytorch docs.')
    parser.add_argument('--weight_decay', type=float, help='The weight decay '
                        'for the learning rate.')
    parser.add_argument('--amsgrad', type=bool, help='The amsgrad option for '
                        'the optimizer.')
    parser.add_argument('--momentum', type=float, help='Momentum option for  '
                        'the optimizer.')

    # learning rate scheduler flags
    parser.add_argument('--lr_scheduler', default='steplr', type=str,
                        choices=['steplr'], help='The learning rate scheduler '
                        'for the optimizer.')
    parser.add_argument('--step_size', type=int, help='The step size option '
                        'for the learning rate scheduler. See pytorch docs.')
    parser.add_argument('--gamma', type=float, help='The gamma option for the '
                        'learing rate scheduler. See pytorch docs.')
    parser.add_argument('--last_epoch', type=int, help='The last_epoch option '
                        'for the learning rate scheduler. See pytorch docs.')

    # saving and loading flags
    parser.add_argument('--num_checkpoints', default=10, type=int,
                        help='Number of checkpoints to maintain for the '
                        'learning process.')
    parser.add_argument('--resume', action='store_true', help='Whether to '
                        'resume the learning from the most recent checkpoint.')
    parser.add_argument('--checkpoint_interval', default=100, type=int,
                        help='Number of steps between saving checkpoints.')
    parser.add_argument('--load_checkpoint', type=str, help='Path to the '
                        'checkpoint file which is to be loaded to warm start '
                        'the training. This option cannot be used if the '
                        '--resume option is used as the resume option by '
                        'default loads the latest_checkpoint.tar file.')

    # knowledge transfer flags
    parser.add_argument('--pretrained_model', type=str, help='Path to the '
                        'pre-trained model that is used for knowledge '
                        'transfer.')

    # model flags
    parser.add_argument('--conv_config', type=eval_str, help='The config of '
                        'the convolutional network ')
    parser.add_argument('--memory_embed', type=int, help='The size of the '
                        'memory vectors')
    parser.add_argument('--num_classes', type=int, help='The number of '
                        'classes in out dataset')
    parser.add_argument('--read_heads', type=int, default=8, help='The number '
                        'of read heads in our controller')
    parser.add_argument('--write_heads', type=int, default=8, help='The number'
                        ' of write heads in our controller')
    parser.add_argument('--gamma', type=float, default=0.99, help='The '
                        'interpolation coefficient for usage weights')
    parser.add_argument('--input_size', type=tuple, default=(28, 28),
                        help='The size of the input images')
    parser.add_argument('--num_entries', default=5, type=int, help='The '
                        'number of tasks we are learning in the continual '
                        'learning framework')
    parser.add_argument('--num_classes', type=int, default=5,
                        help='The number of classes in each task')

    # parse the arguments
    args = parser.parse_args()

    # device for experimenting on.
    args.device = torch.device('cuda' if args.use_cuda else 'cpu')

    # faster loading in GPU with pinned memories
    args.pin_memory = args.use_cuda or args.pin_memory

    # make sure the option is in lower case
    args.optimizer = (args.optimizer).lower()
    args.lr_scheduler = (args.lr_scheduler).lower()

    # get the directory for storing the summary logs
    args.summary_dir = get_summary_dir(args)

    # additional dependent parameters
    feat_size = args.input_size
    for _, _, kernel_size in args.conv_config:
        feat_size = (
            (feat_size[0] - kernel_size) + 1,
            (feat_size[1] - kernel_size) + 1
        )
    args.feat_size = feat_size

    # run the main function
    main(args)
