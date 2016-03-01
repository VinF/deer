#! /usr/bin/env python
"""This script handles reading command line arguments and it starts the
environment, agent and experiment.
"""


import os
import argparse
import logging
import numpy as np

from agent import NeuralAgent
import experiment.base_controllers as bc
from base_classes import Environment, QNetwork

def process_args(args, defaults, description):
    """
    Handle the command line.

    args     - list of command line arguments (not including executable name)
    defaults - a name space with variables corresponding to each of
               the required default command line values.
    description - a string to display at the top of the help message.
    """
    parser = argparse.ArgumentParser(description=description)
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


    
    parser.add_argument('-env', '--env-path', dest="env_name",
                        type=str, default=defaults.ENV_NAME,
                        help='environment_file (default: %(default)s)')

    parser.add_argument('--frame-skip', dest="frame_skip",
                        default=defaults.FRAME_SKIP, type=int,
                        help='Every how many frames to process '
                        '(default: %(default)s)')
#    parser.add_argument('--repeat-action-probability',
#                        dest="repeat_action_probability",
#                        default=defaults.REPEAT_ACTION_PROBABILITY, type=float,
#                        help=('Probability that action choice will be ' +
#                              'ignored (default: %(default)s)'))


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
    parser.add_argument('--network-type', dest="network_type",
                        type=str, default=defaults.NETWORK_TYPE,
                        help=('General_DQN_0' +
                              '|linear (default: %(default)s)'))
    parser.add_argument('--freeze-interval', dest="freeze_interval",
                        type=int, default=defaults.FREEZE_INTERVAL,
                        help=('Interval between target freezes. ' +
                              '(default: %(default)s)'))
    parser.add_argument('--update-frequency', dest="update_frequency",
                        type=int, default=defaults.UPDATE_FREQUENCY,
                        help=('Number of actions before each SGD update. '+
                              '(default: %(default)s)'))
    parser.add_argument('--nn-file', dest="nn_file", type=str, default=None,
                        help='Pickle file containing trained net.')
    parser.add_argument('--deterministic', dest="deterministic",
                        type=bool, default=defaults.DETERMINISTIC,
                        help=('Whether to use deterministic parameters ' +
                              'for learning. (default: %(default)s)'))


    parameters = parser.parse_args(args)


    return parameters


def launch(args, defaults, description):
    """
    Execute a complete run with new API.
    """
    import q_network
    import theano

    logging.basicConfig(level=logging.INFO)
    parameters = process_args(args, defaults, description)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # Instantiate environment
    env = __import__(parameters.env_name).MyEnv(rng)
    if not isinstance(env, Environment):
        raise TypeError("The supplied environment does not subclass base_classes.Environment")

    # Instantiate qnetwork
    qnetwork = q_network.MyQNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_delta,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.network_type,
        parameters.update_rule,
        parameters.batch_accumulator,
        rng)
    if not isinstance(qnetwork, QNetwork):
        raise TypeError("The supplied q-network does not subclass base_classes.QNetwork")
    
    # Instantiate agent
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.batchDimensions()[0]),
        parameters.batch_size,
        parameters.frame_skip,
        rng)

    # Bind controllers to the agent
    agent.attach(bc.VerboseController())
    agent.attach(bc.TrainerController(periodicity=parameters.update_frequency))
    agent.attach(bc.LearningRateController(parameters.learning_rate, parameters.learning_rate_decay))
    agent.attach(bc.DiscountFactorController(parameters.discount, parameters.discount_inc, parameters.discount_max))
    agent.attach(bc.EpsilonController(parameters.epsilon_start, parameters.epsilon_decay, parameters.epsilon_min))
    agent.attach(bc.InterleavedTestEpochController(0, parameters.steps_per_test, [0, 1, 2, 3, 4], summarizeEvery=parameters.period_btw_summary_perfs))
    #agent.attach(bc.InterleavedTestEpochController(0, parameters.steps_per_test, [0, 1, 2, 3, 4, 6], summarizeEvery=parameters.period_btw_summary_perfs))
    #agent.attach(bc.InterleavedTestEpochController(1, parameters.steps_per_test, [0, 1, 2, 3, 4, 5], summarizeEvery=parameters.period_btw_summary_perfs))
    
    # Run the experiment
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    #agent.run(1, parameters.steps_per_epoch)

def testQNetworkAPIUse(envModule):
    import unittests as ut
    rng = np.random.RandomState(0)

    # Instantiate environment
    env = __import__(envModule).MyEnv(rng)
    if not isinstance(env, Environment):
        raise TypeError("The supplied environment does not subclass base_classes.Environment")

    # Instantiate qnetwork
    qnetwork = ut.MyQNetwork(env, 10)
    
    # Instantiate agent
    agent = NeuralAgent(
        env,
        qnetwork,
        150,
        max(env.batchDimensions()[0]),
        10,
        1,
        rng)

    # Bind controllers to the agent
    agent.attach(bc.TrainerController(periodicity=10))
    agent.attach(bc.LearningRateController(0.5, 0.1))
    agent.attach(bc.DiscountFactorController(0.2, 0.1))
    agent.attach(bc.EpsilonController(0.9, 0.1, 0.2))
    agent.attach(bc.InterleavedTestEpochController(20, [0, 1, 2, 3], summarizeEvery=20))
    
    # Run the experiment
    agent.run(5, 64)


if __name__ == '__main__':
    testQNetworkAPIUse("Toy_env")
