"""ALE launcher. See Wiki for more details about this experiment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from default_parser import process_args
from agent_ale import ALEAgent
from q_networks.q_net_theano import MyQNetwork
from environments import ALE_env
import experiment.base_controllers as bc

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 250000
    EPOCHS = 40
    STEPS_PER_TEST = 125000
    PERIOD_BTW_SUMMARY_PERFS = 1
    
    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 4

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY = 0.99
    DISCOUNT = 0.95
    DISCOUNT_INC = 0.99
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = .1
    EPSILON_DECAY = 100000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 10000
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
    env = ALE_env(rng, frame_skip=parameters.frame_skip, 
                ale_options=[{"key": "random_seed", "value": rng.randint(9999)}, 
                             {"key": "color_averaging", "value": True},
                             {"key": "repeat_action_probability", "value": 0.}])

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
    agent = ALEAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    # --- Bind controllers to the agent ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "ALE_" + h
    print("The parameters hash is: {}".format(h))
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
        validationID=ALE_env.VALIDATION_MODE,
        testID=None,
        unique_fname=fname))

    agent.attach(bc.InterleavedTestEpochController(
        id=ALE_env.VALIDATION_MODE, 
        epochLength=parameters.steps_per_test,
        controllersToDisable=[0, 1, 2, 3, 4],
        periodicity=2,
        showScore=True,
        summarizeEvery=1))
    
    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    
    # --- Show results ---
    basename = "scores/" + fname
    scores = joblib.load(basename + "_scores.jldump")
    plt.plot(range(1, len(scores['vs'])+1), scores['vs'], label="VS", color='b')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.savefig(basename + "_scores.pdf")
    plt.show()
