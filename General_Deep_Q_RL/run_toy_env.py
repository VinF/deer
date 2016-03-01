#! /usr/bin/env python
"""
Execute a training run of general deep-Q-Leaning with following parameters:
"""

import launcher
import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from agent import NeuralAgent
from q_network import MyQNetwork
from Toy_env import MyEnv
import experiment.base_controllers as bc

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 1000
    EPOCHS = 50
    STEPS_PER_TEST = 500
    PERIOD_BTW_SUMMARY_PERFS = 10

    # ----------------------
    # Environment Parameters
    # ----------------------
    ENV_NAME = "Toy_env"
    FRAME_SKIP = 1#4

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = 0.0002
    LEARNING_RATE_DECAY = 1.
    DISCOUNT = 0.9
    DISCOUNT_INC = 1.
    DISCOUNT_MAX = 0.99

    RMS_DECAY = 0.9#.99 # (Rho)
    RMS_EPSILON = 0.0001#.01
    MOMENTUM = 0 # Note that the "momentum" value mentioned in the Nature
                 # paper is not used in the same way as a traditional momentum
                 # term.  It is used to track gradient for the purpose of
                 # estimating the standard deviation. This package uses
                 # rho/RMS_DECAY to track both the history of the gradient
                 # and the squared gradient.
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = True
    CUDNN_DETERMINISTIC = False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parameters = launcher.process_args(sys.argv[1:], Defaults, __doc__)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # Instantiate environment
    env = MyEnv(rng)

    # Instantiate qnetwork
    qnetwork = MyQNetwork(
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
    
    # Instantiate agent
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.batchDimensions()[i][0] for i in range(len(env.batchDimensions()))),
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
        
    # Run the experiment
    agent.run(parameters.epochs, parameters.steps_per_epoch)
