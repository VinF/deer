"""2-Storage Microgrid launcher. See Wiki for more details about this experiment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from default_parser import process_args
from agent import NeuralAgent
from q_networks.q_net_theano import MyQNetwork
from environments import MG_two_storages_env
import experiment.base_controllers as bc

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 365*24-1
    EPOCHS = 200
    STEPS_PER_TEST = 365*24-1
    PERIOD_BTW_SUMMARY_PERFS = -1  # Set to -1 for avoiding call to env.summarizePerformance
    
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
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.9
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.98
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .3
    EPSILON_DECAY = 500000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = True




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # Instantiate environment
    env = MG_two_storages_env(rng)

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
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    # Bind controllers to the agent
    VALIDATION_MODE = 0
    TEST_MODE = 1
    fname = hash(vars(parameters), hash_name="sha1")
    print("The parameters hash is: {}".format(fname))
    print("The parameters are: {}".format(parameters))

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

    agent.attach(bc.FindBestController(
        validationID=VALIDATION_MODE, 
        testID=TEST_MODE,
        unique_fname=fname, 
        showPlot=False))
    
    agent.attach(bc.InterleavedTestEpochController(
        id=VALIDATION_MODE, 
        epochLength=parameters.steps_per_test, 
        controllersToDisable=[0, 1, 2, 3, 4, 7], 
        periodicity=2, 
        showScore=True,
        summarizeEvery=-1))

    agent.attach(bc.InterleavedTestEpochController(
        id=TEST_MODE,
        epochLength=parameters.steps_per_test,
        controllersToDisable=[0, 1, 2, 3, 4, 6],
        periodicity=2,
        showScore=True,
        summarizeEvery=parameters.period_btw_summary_perfs))
    
    # Run the experiment
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.run(parameters.epochs, parameters.steps_per_epoch)
