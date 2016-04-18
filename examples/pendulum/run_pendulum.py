""" Pendulum environment launcher.
Same principles as run_toy_env. See the wiki for more details.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np

import deer.experiment.base_controllers as bc
from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.q_networks.q_net_theano import MyQNetwork
from pendulum_env import MyEnv as pendulum_env

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 1000
    EPOCHS = 200
    STEPS_PER_TEST = 1000
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
    LEARNING_RATE = 0.001
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = .99
    DISCOUNT_MAX = 0.95
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .2
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 100
    DETERMINISTIC = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = pendulum_env(rng)

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
    # For comments, please refer to run_toy_env.py
    agent.attach(bc.VerboseController(
        evaluateOn='epoch', 
        periodicity=1))

    agent.attach(bc.TrainerController(
        evaluateOn='action', 
        periodicity=parameters.update_frequency, 
        showEpisodeAvgVValue=True, 
        showAvgBellmanResidual=True))

    agent.attach(bc.LearningRateController(
        initialLearningRate=parameters.learning_rate,
        learningRateDecay=parameters.learning_rate_decay,
        periodicity=1))

    agent.attach(bc.DiscountFactorController(
        initialDiscountFactor=parameters.discount,
        discountFactorGrowth=parameters.discount_inc,
        discountFactorMax=parameters.discount_max,
        periodicity=1))

    agent.attach(bc.EpsilonController(
        initialE=parameters.epsilon_start, 
        eDecays=parameters.epsilon_decay, 
        eMin=parameters.epsilon_min,
        evaluateOn='action', 
        periodicity=1, 
        resetEvery='none'))

    agent.attach(bc.InterleavedTestEpochController(
        id=0, 
        epochLength=parameters.steps_per_test, 
        controllersToDisable=[0, 1, 2, 3, 4], 
        periodicity=2, 
        showScore=True,
        summarizeEvery=parameters.period_btw_summary_perfs))
    
    # --- Run the experiment ---
    agent.run(parameters.epochs, parameters.steps_per_epoch)
