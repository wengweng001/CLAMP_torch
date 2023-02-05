import argparse
from cProfile import run
import random
import numpy as np
import torch

from train_main import run_main
import approach

from utils import get_model_names
from networks import allmodels
import os
import warnings
warnings.filterwarnings("ignore")

wandb=None

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def main(parser, args):
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    run_main(parser, args)



if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="CLAMP")
    ########################General#########################

    parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
    parser.add_argument('--data-dir', type=str, default='./data', dest='d_dir', help='default: %(default)s')
    parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help='default: %(default)s')
    parser.add_argument('--model-dir', type=str, default='./models', dest='m_dir', help='default: %(default)s')
    parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help='default: %(default)s')
    parser.add_argument('--log', default=['disk'], type=str, choices=['disk', 'tensorboard'],
                        help='Loggers used (disk, tensorboard) (default=%(default)s)', nargs='*', metavar="LOGGER")

    parser.add_argument('--runs', type=int, default=5, help='how often to repeat?')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--no-gpus', action='store_false', dest='gpu', help="don't use GPUs")
    parser.add_argument('--scenario', type=str, default='class', choices=['task', 'class', 'domain'], help="only class incremental is available now")
    parser.add_argument('--tasks', default=5, type=int, help='number of tasks')
    parser.add_argument('--stop-at-task', default=99, type=int, required=False,
                        help='Stop training after specified task (default=%(default)s)')
    parser.add_argument('--approach', default='finetuning', type=str, choices=approach.__all__,
                        help='Learning approach used (default=%(default)s)', metavar="APPROACH")
    parser.add_argument('--analysis', action='store_true', help="visulize tSNE plots")

    parser.add_argument('--source', default='splitMNIST', choices=['splitMNIST', 'splitUSPS', 'Office-31', 'Office-Home'])
    parser.add_argument('--target', default='splitUSPS', choices=['splitUSPS', 'splitMNIST', 'Office-31', 'Office-Home'])

    # parser.add_argument('--lstm-layers', type=int, default=1, dest='lstm_lay', help='# of LSTM layers')
    # parser.add_argument('--lstm-units', type=int, metavar='N', help='# of units in LSTM')
    parser.add_argument('--fc-units', default=100, type=int, metavar='N', help='# of units in first fc-layers')
    parser.add_argument('--fc-drop', type=float, default=0., help='dropout probability for fc-units')
    parser.add_argument('--fc-bn', type=str, default='no', help='use batch-norm in the fc-layers (no|yes)')
    parser.add_argument('--fc-nl', type=str, default='relu', choices=['relu', 'leakyrelu'])
    parser.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                    " (instead of a 'multi-headed' one) ")

    parser.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")

    parser.add_argument('--lr-bm', dest='lr1', type=float, default=0.001, help='learning rate of base model')
    parser.add_argument('--lr-a', dest='lr2', type=float, default=0.001, help='learning rate of assessor')
    parser.add_argument('--wd',  type=float, default=0.0005, help='optimizer wight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='optimizer momentum')
    parser.add_argument('--batch', type=int, default=128, help='batch-size')
    parser.add_argument('--batch-d', type=int, default=64, help='batch-size-domain-adaptation')
    parser.add_argument('--optimizer', type=str, choices=['sgd'], default='sgd')
    
    parser.add_argument('--warmup-nepochs', default=1, type=int, required=False,
                        help='Number of warm-up epochs (default=%(default)s)')
    parser.add_argument('--warmup-lr-factor', default=1.0, type=float, required=False,
                        help='Warm-up learning rate factor (default=%(default)s)')

    parser.add_argument('--lr-min', default=1e-4, type=float, required=False,
                        help='Minimum learning rate (default=%(default)s)')
    parser.add_argument('--lr-factor', default=3, type=float, required=False,
                        help='Learning rate decreasing factor (default=%(default)s)')
    parser.add_argument('--lr-patience', default=5, type=int, required=False,
                        help='Maximum patience to wait before decreasing learning rate (default=%(default)s)')
    parser.add_argument('--clipping', default=10000, type=float, required=False,
                        help='Clip gradient norm (default=%(default)s)')

    # model parameters
    parser.add_argument('--network', default='resnet18',
                        choices=get_model_names(),
                        help='backbone architecture: ' +
                             ' | '.join(get_model_names()) +
                             ' (default: resnet18)')
    # parser.add_argument('--network', default='resnet32', type=str, choices=allmodels,
    #                     help='Network architecture used (default=%(default)s)', metavar="NETWORK")
    parser.add_argument('--keep-existing-head', action='store_true',
                        help='Disable removing classifier last layer (default=%(default)s)')
    parser.add_argument('--pretrained', action='store_true',
                        help='Use pretrained backbone (default=%(default)s)')

    parser.add_argument('--multi-softmax', action='store_true',
                        help='Apply separate softmax for each task (default=%(default)s)')
    parser.add_argument('--fix-bn', action='store_true',
                        help='Fix batch normalization after first task (default=%(default)s)')

    parser.add_argument('--epoch', type=int, default=1, dest="epoch", help="meta learning iterations")
    parser.add_argument('--epoch-inner', type=int, default=1, dest="epochIn", help="meta learning iterations")
    parser.add_argument('--epoch-outer', type=int, default=1, dest="epochOut", help="meta learning iterations")

    parser.add_argument('--no-cudnn-deterministic', action='store_true',
                        help='Disable CUDNN deterministic (default=%(default)s)')

    parser.add_argument('--eval-on-train', action='store_true',
                        help='Show train loss and accuracy (default=%(default)s)')
    parser.add_argument('--save-models', action='store_true',
                        help='Save trained models (default=%(default)s)')

    # domain adaptation strategies
    parser.add_argument('--no-pool', action='store_true',
                        help='no pool layer after the feature extractor.')
    parser.add_argument('--scratch', action='store_true', help='whether train from scratch.')

    # gridsearch args
    parser.add_argument('--gridsearch-tasks', default=-1, type=int,
                        help='Number of tasks to apply GridSearch (-1: all tasks) (default=%(default)s)')

    # CLAMP args
    parser.add_argument('--pseudo', action='store_true',
                        help='Use pseudo label strategy (default=%(default)s)')
    parser.add_argument('--meta', action='store_true',
                        help='Use meta learning (default=%(default)s)')
    parser.add_argument('--domain', action='store_true',
                        help='Use DANN domain invariant strategy (default=%(default)s)')
    parser.add_argument('--alpha', default=1.0, type=float,
                        help='tuning factor for DER loss and distilation loss (default=%(default)s)')

    # exemplars args
    parser.add_argument('--num-exemplars1', default=0, type=int, required=False,
                        help='Fixed memory, total number of source domain exemplars (default=%(default)s)')
    parser.add_argument('--num-exemplars-per-class1', default=0, type=int, required=False,
                        help='Growing memory, number of source domain exemplars per class (default=%(default)s)')
    parser.add_argument('--num-exemplars2', default=0, type=int, required=False,
                        help='Fixed memory, total number of target domain exemplars (default=%(default)s)')
    parser.add_argument('--num-exemplars-per-class2', default=0, type=int, required=False,
                        help='Growing memory, number of target domain exemplars per class (default=%(default)s)')

    args = parser.parse_args()
    if not args.gpu:
        args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        args.device = torch.device('cpu')
    main(parser, args)