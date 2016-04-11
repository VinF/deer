"""This module contains a function to help parse command-line arguments.

Authors: Vincent Francois-Lavet, David Taralla
"""


import argparse

def process_args(args, defaults):
    """Handle the command line and return an object containing all the parameters.

    Arguments:
        args     - list of command line arguments (not including executable name)
        defaults - a name space with variables corresponding to each of the required default command line values.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', dest="epochs", type=int,
                        default=defaults.EPOCHS,
                        help='Number of training epochs (default: %(default)s)')
    parser.add_argument('-s', '--steps-per-epoch', dest="steps_per_epoch",
                        type=int, default=defaults.STEPS_PER_EPOCH,
                        help='Number of steps per epoch (default: %(default)s)')
    parser.add_argument('-t', '--test-length', dest="steps_per_test",
                        type=int, default=defaults.STEPS_PER_TEST,
                        help='Number of steps per test (default: %(default)s)')
    parser.add_argument('-f', '--freq_summary_perfs', dest="period_btw_summary_perfs",
                        type=int, default=defaults.PERIOD_BTW_SUMMARY_PERFS,
                        help='freq summary perfs (default: %(default)s)')
    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')
    parser.add_argument('--update-rule', dest="update_rule",
                        type=str, default=defaults.UPDATE_RULE,
                        help=('deepmind_rmsprop|rmsprop|sgd ' +
                              '(default: %(default)s)'))
    parser.add_argument('--batch-accumulator', dest="batch_accumulator",
                        type=str, default=defaults.BATCH_ACCUMULATOR,
                        help=('sum|mean (default: %(default)s)'))
    parser.add_argument('--learning-rate', dest="learning_rate",
                        type=float, default=defaults.LEARNING_RATE,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--learning-rate-decay', dest="learning_rate_decay",
                        type=float, default=defaults.LEARNING_RATE_DECAY,
                        help='Learning rate (default: %(default)s)')
    parser.add_argument('--rms-decay', dest="rms_decay",
                        type=float, default=defaults.RMS_DECAY,
                        help='Decay rate for rms_prop (default: %(default)s)')
    parser.add_argument('--rms-epsilon', dest="rms_epsilon",
                        type=float, default=defaults.RMS_EPSILON,
                        help='Denominator epsilson for rms_prop ' +
                        '(default: %(default)s)')
    parser.add_argument('--momentum', type=float, default=defaults.MOMENTUM,
                        help=('Momentum term for Nesterov momentum. '+
                              '(default: %(default)s)'))
    parser.add_argument('--clip-delta', dest="clip_delta", type=float,
                        default=defaults.CLIP_DELTA,
                        help=('Max absolute value for Q-update delta value. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--discount', type=float, default=defaults.DISCOUNT,
                        help='Discount rate init')
    parser.add_argument('--discount_inc', type=float, default=defaults.DISCOUNT_INC,
                        help='Discount rate')
    parser.add_argument('--discount_max', type=float, default=defaults.DISCOUNT_MAX,
                        help='Discount rate max')
    parser.add_argument('--epsilon-start', dest="epsilon_start",
                        type=float, default=defaults.EPSILON_START,
                        help=('Starting value for epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--epsilon-min', dest="epsilon_min",
                        type=float, default=defaults.EPSILON_MIN,
                        help='Minimum epsilon. (default: %(default)s)')
    parser.add_argument('--epsilon-decay', dest="epsilon_decay",
                        type=float, default=defaults.EPSILON_DECAY,
                        help=('Number of steps to minimum epsilon. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--max-history', dest="replay_memory_size",
                        type=int, default=defaults.REPLAY_MEMORY_SIZE,
                        help=('Maximum number of steps stored in replay ' +
                              'memory. (default: %(default)s)'))
    parser.add_argument('--batch-size', dest="batch_size",
                        type=int, default=defaults.BATCH_SIZE,
                        help='Batch size. (default: %(default)s)')
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    parser.add_argument('--network-type', dest="network_type",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('General_DQN_0' +
                              '|linear (default: %(default)s)'))
    parser.add_argument('--deterministic', dest='deterministic', action='store_true',
                        help=('If fixed seed (default: %(default)s)'))
    parser.add_argument('--no-deterministic', dest='deterministic', action='store_false',
                        help=('If no fixed seed'))
    parser.set_defaults(deterministic=True)

    parameters = parser.parse_args(args)

    return parameters

if __name__ == '__main__':
    pass
