import argparse
import os
from model import *
import main
import utils


parser = argparse.ArgumentParser()

parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--data-dir', type=str, default='./datasets', dest='d_dir', help='default: %(default)s')
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help='default: %(default)s')
parser.add_argument('--results-dir', type=str, default='./results', dest='r_dir', help='default: %(default)s')

parser.add_argument('--runs', type=int, default=5, help='how often to repeat?')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--no-gpus', action='store_false', dest='gpu', help="don't use GPUs")
parser.add_argument('--scenario', type=str, default='class', choices=['task', 'class', 'domain'])
parser.add_argument('--tasks', type=int, help='number of tasks')

parser.add_argument('--source', default='splitMNIST', choices=['splitMNIST',])
parser.add_argument('--target', default='splitUSPS', choices=['splitUSPS',])

parser.add_argument('--lstm-layers', type=int, default=1, dest='lstm_lay', help='# of LSTM layers')
parser.add_argument('--lstm-units', type=int, metavar='N', help='# of units in LSTM')
parser.add_argument('--fc-units', type=int, metavar='N', help='# of units in first fc-layers')
parser.add_argument('--fc-drop', type=float, default=0., help='dropout probability for fc-units')
parser.add_argument('--fc-bn', type=str, default='no', help='use batch-norm in the fc-layers (no|yes)')
parser.add_argument('--fc-nl', type=str, default='relu', choices=['relu', 'leakyrelu'])
parser.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                " (instead of a 'multi-headed' one) ")

parser.add_argument('--budget', type=int, default=1000, dest="budget", help="how many samples can be stored per task?")
parser.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")

parser.add_argument('--lr-bm', dest='lr1', type=float, default=0.01, help='learning rate of base model')
parser.add_argument('--lr-a', dest='lr2', type=float, default=0.01, help='learning rate of assessor')
parser.add_argument('--batch', type=int, default=128, help='batch-size')
parser.add_argument('--optimizer', type=str, choices=['sgd'], default='sgd')


parser.add_argument('--alpha-pa', type=float, dest="alpha1", help="regularization term for process adaptation loss")
parser.add_argument('--alpha-ps', type=float, default=0.8, dest="alpha2", help="pseudo label threshold")
parser.add_argument('--alpha-tg', type=float, dest="alpha3", help="regularization term for target process loss")
parser.add_argument('--alpha-ed', type=float, dest="alpha4", help="regularization term for Euclidean distance")
parser.add_argument('--alpha-der', type=float, dest="alpha5", help="regularization term for dark experience replay")
parser.add_argument('--epoch', type=int, default=100, dest="epoch", help="meta learning iterations")




def get_results(args):
    # -get param-stamp
    param_stamp = get_param_stamp_from_args(args)
    # -check whether already run; if not do so
    if os.path.isfile("{}/dict-{}.pkl".format(args.r_dir, param_stamp)):
        print("{}: already run".format(param_stamp))
    else:
        print("{}: ...running...".format(param_stamp))
        main.run(args)
    # -get results-dict
    dict = utils.load_object("{}/dict-{}".format(args.r_dir, param_stamp))
    # -get average precision
    fileName = '{}/prec-{}.txt'.format(args.r_dir, param_stamp)
    file = open(fileName)
    ave = float(file.readline())
    file.close()
    # -print average precision on screen
    print("--> average precision: {}".format(ave))
    # -return average precision
    return (dict, ave)


def collect_all(method_dict, seed_list, args, name=None):
    # -print name of method on screen
    if name is not None:
        print("\n------{}------".format(name))
    # -run method for all random seeds
    for seed in seed_list:
        args.seed = seed
        method_dict[seed] = get_results(args)
    # -return updated dictionary with results
    return method_dict



if __name__ == '__main__':

    ## Load input-arguments
    args = parser.parse_args()
    # # -set default-values for certain arguments based on chosen scenario & experiment
    # args = set_default_values(args)

    #-------------------------------------------------------------------------------------------------#

    #--------------------------#
    #----- RUN ALL MODELS -----#
    #--------------------------#

    seed_list = list(range(args.seed, args.seed+args.n_seeds))


    ###----"BASELINES"----###

    ## Offline
    args.add_exemplars = True
    OFF = {}
    OFF = collect_all(OFF, seed_list, args, name="Offline")

    ## None
    args.add_exemplars = False
    NONE = {}
    NONE = collect_all(NONE, seed_list, args, name="None")


