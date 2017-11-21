"""2-Storage Microgrid launcher. See the docs for more details about this experiment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
from joblib import hash, dump, load
import os
import matplotlib.pyplot as plt

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.q_networks.q_net_theano import MyQNetwork
from MG_two_storages_env import MyEnv as MG_two_storages_env
import deer.experiment.base_controllers as bc

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
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0005
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
        parameters.update_rule,
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
        evaluate_on='epoch',
        periodicity=1))
    
    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency,
        show_episode_avg_V_value=True,
        show_avg_Bellman_residual=True))
    
    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we 
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))
    
    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))
    
    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start,
        e_decays=parameters.epsilon_decay,
        e_min=parameters.epsilon_min,
        evaluate_on='action',
        periodicity=1,
        reset_every='none'))

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
        epoch_length=parameters.steps_per_test,
        controllers_to_disable=[0, 1, 2, 3, 4, 7],
        periodicity=2, 
        show_score=True,
        summarize_every=-1))
    
    # Besides inserting a validation epoch (required if one wants to find the best neural network over all training
    # epochs), we also wish to interleave a "test epoch" between each training epoch ("one of two epochs", hence the 
    # periodicity=2). We do not want these test epoch to interfere with the training of the agent nor with its 
    # validation. Therefore, we will disable these controllers for the whole duration of the test epochs interleaved 
    # this way, using the controllersToDisable argument of the InterleavedTestEpochController. For each test epoch, we 
    # want also to display the sum of all rewards obtained, hence the showScore=True. Finally, we want to call the 
    # summarizePerformance method of MG_two_storage_env every [parameters.period_btw_summary_perfs] *test* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=MG_two_storages_env.TEST_MODE,
        epoch_length=parameters.steps_per_test,
        controllers_to_disable=[0, 1, 2, 3, 4, 6],
        periodicity=2,
        show_score=True,
        summarize_every=parameters.period_btw_summary_perfs))
    
    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    
    # --- Show results ---
    basename = "scores/" + fname
    scores = load(basename + "_scores.jldump")
    plt.plot(range(1, len(scores['vs'])+1), scores['vs'], label="VS", color='b')
    plt.plot(range(1, len(scores['ts'])+1), scores['ts'], label="TS", color='r')
    plt.legend()
    plt.xlabel("Number of epochs")
    plt.ylabel("Score")
    plt.savefig(basename + "_scores.pdf")
    plt.show()
