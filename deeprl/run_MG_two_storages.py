"""2-Storage Microgrid launcher. See Wiki for more details about this experiment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os
import matplotlib.pyplot as plt

from deeprl.default_parser import process_args
from deeprl.agent import NeuralAgent
from deeprl.q_networks.q_net_theano import MyQNetwork
from deeprl.environments import MG_two_storages_env
import deeprl.experiment.base_controllers as bc

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
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = MG_two_storages_env(rng)

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
    
    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "MG2S_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

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

    # We wish to discover, among all versions of our neural network (i.e., after every training epoch), which one 
    # seems to generalize the better, thus which one has the highest validation score. However we also want to keep 
    # track of a "true generalization score", the "test score". Indeed, what if we overfit the validation score ?
    # To achieve these goals, one can use the FindBestController along two InterleavedTestEpochControllers, one for
    # each mode (validation and test). It is important that the validationID and testID are the same than the id 
    # argument of the two InterleavedTestEpochControllers (implementing the validation mode and test mode 
    # respectively). The FindBestController will dump on disk the validation and test scores for each and every 
    # network, as well as the structure of the neural network having the best validation score. These dumps can then
    # used to plot the evolution of the validation and test scores (see below) or simply recover the resulting neural 
    # network for your application.
    agent.attach(bc.FindBestController(
        validationID=MG_two_storages_env.VALIDATION_MODE, 
        testID=MG_two_storages_env.TEST_MODE,
        unique_fname=fname))
    
    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want 
    # these validation epoch to interfere with the training of the agent, which is well established by the 
    # TrainerController, EpsilonController and alike, nor with its testing (see next controller). Therefore, we will 
    # disable these controllers for the whole duration of the validation epochs interleaved this way, using the 
    # controllersToDisable argument of the InterleavedTestEpochController. For each validation epoch, we want also to 
    # display the sum of all rewards obtained, hence the showScore=True. Finally, we never want this controller to call 
    # the summarizePerformance method of MG_two_storage_env.
    agent.attach(bc.InterleavedTestEpochController(
        id=MG_two_storages_env.VALIDATION_MODE, 
        epochLength=parameters.steps_per_test, 
        controllersToDisable=[0, 1, 2, 3, 4, 7], 
        periodicity=2, 
        showScore=True,
        summarizeEvery=-1))
    
    # Besides inserting a validation epoch (required if one wants to find the best neural network over all training
    # epochs), we also wish to interleave a "test epoch" between each training epoch ("one of two epochs", hence the 
    # periodicity=2). We do not want these test epoch to interfere with the training of the agent nor with its 
    # validation. Therefore, we will disable these controllers for the whole duration of the test epochs interleaved 
    # this way, using the controllersToDisable argument of the InterleavedTestEpochController. For each test epoch, we 
    # want also to display the sum of all rewards obtained, hence the showScore=True. Finally, we want to call the 
    # summarizePerformance method of MG_two_storage_env every [parameters.period_btw_summary_perfs] *test* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=MG_two_storages_env.TEST_MODE,
        epochLength=parameters.steps_per_test,
        controllersToDisable=[0, 1, 2, 3, 4, 6],
        periodicity=2,
        showScore=True,
        summarizeEvery=parameters.period_btw_summary_perfs))
    
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
    plt.plot(range(1, len(scores['ts'])+1), scores['ts'], label="TS", color='r')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.savefig(basename + "_scores.pdf")
    plt.show()
