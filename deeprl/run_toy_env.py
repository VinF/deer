"""Toy environment launcher. See the wiki for more details about this environment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from deeprl.default_parser import process_args
from deeprl.agent import NeuralAgent
from deeprl.q_networks.q_net_theano import MyQNetwork
from deeprl.environments import Toy_env
import deeprl.experiment.base_controllers as bc

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
    FRAME_SKIP = 1

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
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001#.01
    MOMENTUM = 0
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = Toy_env(rng)

    # --- Instantiate qnetwork ---
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
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluateOn='epoch', 
        periodicity=1))

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluateOn='action', 
        periodicity=parameters.update_frequency, 
        showEpisodeAvgVValue=True, 
        showAvgBellmanResidual=True))

    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we 
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initialLearningRate=parameters.learning_rate,
        learningRateDecay=parameters.learning_rate_decay,
        periodicity=1))

    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initialDiscountFactor=parameters.discount,
        discountFactorGrowth=parameters.discount_inc,
        discountFactorMax=parameters.discount_max,
        periodicity=1))

    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(bc.EpsilonController(
        initialE=parameters.epsilon_start, 
        eDecays=parameters.epsilon_decay, 
        eMin=parameters.epsilon_min,
        evaluateOn='action', 
        periodicity=1, 
        resetEvery='none'))

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "test epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want these 
    # test epoch to interfere with the training of the agent, which is well established by the TrainerController, 
    # EpsilonController and alike. Therefore, we will disable these controllers for the whole duration of the test 
    # epochs interleaved this way, using the controllersToDisable argument of the InterleavedTestEpochController. 
    # The value of this argument is a list of the indexes of all controllers to disable, their index reflecting in 
    # which order they were added. Here, "0" is refering to the firstly attached controller, thus the 
    # VerboseController; "2" refers to the thirdly attached controller, thus the LearningRateController; etc. The order 
    # in which the indexes are listed is not important.
    # For each test epoch, we want also to display the sum of all rewards obtained, hence the showScore=True.
    # Finally, we want to call the summarizePerformance method of Toy_Env every [parameters.period_btw_summary_perfs]
    # *test* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=0, 
        epochLength=parameters.steps_per_test, 
        controllersToDisable=[0, 1, 2, 3, 4], 
        periodicity=2, 
        showScore=True,
        summarizeEvery=parameters.period_btw_summary_perfs))
        
    # --- Run the experiment ---
    agent.run(parameters.epochs, parameters.steps_per_epoch)
