import os
import argparse

DEFAULT_ARCH_CHOICES = ['mnist']

def make_standard_parser(description='No decription provided', arch_choices=DEFAULT_ARCH_CHOICES):
    '''Make a standard parser, probably good for many experiments.

    Arguments:

      description: just used for help

      arch_choices: list of strings that may be specified when
         selecting architecture type. For example, ('mnist', 'cifar')
         would allow selection of different networks for each
         dataset. architecture may also be toggled via the --conv and
         --xprop switches. Default architecture is the first in the
         list.
    '''
    
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=lambda prog: argparse.ArgumentDefaultsHelpFormatter(prog)
    )

    # Optimization
    parser.add_argument('--opt', type=str, default='sgd', choices=('sgd', 'rmsprop', 'adam'), help='Which optimizer to use')
    parser.add_argument('--lr', '-L', type=float, default=.001, help='learning rate')
    parser.add_argument('--mom', '-M', type=float, default=.9, help='momentum (only has effect for sgd/rmsprop)')
    parser.add_argument('--beta1', type=float, default=.9, help='beta1 for adam opt')
    parser.add_argument('--beta2', type=float, default=.99, help='beta2 for adam opt')
    parser.add_argument('--adameps', type=float, default=1e-8, help='epsilon for adam opt')
    parser.add_argument('--epochs', '-E',type=int, default=5, help='number of epochs.')

    # Model
    parser.add_argument('--arch', type=str, default=arch_choices[0],
                        choices=arch_choices, help='Which architecture to use (choices: %s).' % arch_choices)
    parser.add_argument('--conv', '-C', action='store_true', help='Use a conv model.')
    parser.add_argument('--xprop', '-X', action='store_true', help='Use an xprop model')
    parser.add_argument('--springprop', '-S', action='store_true', help='Use an springprop model')
    parser.add_argument('--springt', '-t', type=float, default=0.5, help='T value to use for springs')
    parser.add_argument('--learncoords', '--lc', action='store_true', help='Learn coordinates (update them during training) instead of keeping them fixed.')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 regularization to apply to direct parameters.')
    parser.add_argument('--l2i', type=float, default=0.0, help='L2 regularization to apply to indirect parameters.')

    # Experimental setup
    parser.add_argument('--seed', type=int, default=0, help='random number seed for intial params and tf graph')
    parser.add_argument('--test', action='store_true', help='Use test data instead of validation data (for final run).')
    parser.add_argument('--shuffletrain', '--st', dest='shuffletrain', action='store_true', help='Shuffle training set each epoch.')
    parser.add_argument('--noshuffletrain', '--nst', dest='shuffletrain', action='store_false', help='Do not shuffle training set each epoch. Ignore the following "default" value:')
    parser.set_defaults(shuffletrain=True)
    
    # Misc
    parser.add_argument('--ipy', '-I', action='store_true', help='drop into embedded iPython for debugging.')
    parser.add_argument('--nocolor', '--nc', action='store_true', help='Do not use color output (for scripts).')
    parser.add_argument('--skipval', action='store_true', help='Skip validation set entirely.')
    parser.add_argument('--verbose', '-V', action='store_true', help='Verbose mode (print some extra stuff)')

    # Saving a loading
    parser.add_argument('--snapshot-to', type=str, default='net', help='Where to snapshot to. --snapshot-to NAME produces NAME_iter.h5 and NAME.json')
    parser.add_argument('--snapshot-every', type=int, default=-1, help='Snapshot every N minibatches. 0 to disable snapshots, -1 to snapshot only on last iteration.')
    parser.add_argument('--load', type=str, default=None, help='Snapshot to load from: specify as H5_FILE:MISC_FILE.')
    parser.add_argument('--output', '-O', type=str, default=None, help='directory output TF results to. If nothing else: skips output.')

    return parser
