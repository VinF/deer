"""Maze launcher

Author: Vincent Francois-Lavet
"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.CRAR_keras import CRAR
from maze_env import MyEnv as maze_env
import deer.experiment.base_controllers as bc

from deer.policies import EpsilonGreedyPolicy


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 250
    STEPS_PER_TEST = 200
    PERIOD_BTW_SUMMARY_PERFS = 1
    
    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 2

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY = 1.#0.995
    DISCOUNT = 0.9
    DISCOUNT_INC = 1
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 1.0
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False

HIGHER_DIM_OBS = True
HIGH_INT_DIM = True
N_SAMPLES=200000
samples_transfer=100


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = maze_env(rng, higher_dim_obs=HIGHER_DIM_OBS)
    
    # --- Instantiate learning_algo ---
    learning_algo = CRAR(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng,
        double_Q=True,
        high_int_dim=HIGH_INT_DIM,
        internal_dim=3,
        div_entrop_loss=1.)
    
    train_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 1.)
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.1)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        learning_algo,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        train_policy=train_policy,
        test_policy=test_policy)

    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "test_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='epoch', 
        periodicity=1))
    
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

    agent.run(1, N_SAMPLES)
    
    #print (agent._dataset._rewards._data[0:500])
    #print (agent._dataset._terminals._data[0:500])
    print("end gathering data")
    old_rewards=agent._dataset._rewards._data
    old_terminals=agent._dataset._terminals._data
    old_actions=agent._dataset._actions._data
    old_observations=agent._dataset._observations[0]._data

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action', 
        periodicity=parameters.update_frequency, 
        show_episode_avg_V_value=True, 
        show_avg_Bellman_residual=True))
    
    # We wish to discover, among all versions of our neural network (i.e., after every training epoch), which one 
    # seems to generalize the better, thus which one has the highest validation score. Here, we do not care about the
    # "true generalization score", or "test score".
    # To achieve this goal, one can use the FindBestController along with an InterleavedTestEpochControllers. It is 
    # important that the validationID is the same than the id argument of the InterleavedTestEpochController.
    # The FindBestController will dump on disk the validation scores for each and every network, as well as the 
    # structure of the neural network having the best validation score. These dumps can then used to plot the evolution 
    # of the validation and test scores (see below) or simply recover the resulting neural network for your 
    # application.
    #agent.attach(bc.FindBestController(
    #    validationID=maze_env.VALIDATION_MODE,
    #    testID=None,
    #    unique_fname=fname))
    
    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want 
    # these validation epoch to interfere with the training of the agent, which is well established by the 
    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole 
    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the 
    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards 
    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every 
    # [parameters.period_btw_summary_perfs] *validation* epochs.
    valid0=bc.InterleavedTestEpochController(
        id=0, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1)
    agent.attach(valid0)

    valid1=bc.InterleavedTestEpochController(
        id=1, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1)
    agent.attach(valid1)

    valid2=bc.InterleavedTestEpochController(
        id=2, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1)
    agent.attach(valid2)
    
    valid3=bc.InterleavedTestEpochController(
        id=3, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1)
    agent.attach(valid3)

    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    agent.gathering_data=False
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    
print (valid0.scores)
print (valid1.scores)
print (valid2.scores)
print (valid3.scores)

#    ###
#    # TRANSFER
#    ###
#    optimized_params=learning_algo.getAllParams()
#    print ("optimized_params")
#    print (optimized_params)
#
#    # --- Instantiate learning_algo ---
##    learning_algo = CRAR(
##        env,
##        parameters.rms_decay,
##        parameters.rms_epsilon,
##        parameters.momentum,
##        parameters.clip_delta,
##        parameters.freeze_interval,
##        parameters.batch_size,
##        parameters.update_rule,
##        rng,
##        double_Q=True,
##        high_int_dim=HIGH_INT_DIM,
##        internal_dim=3)
##    learning_algo.setAllParams(optimized_params)
#
#    rand_ind=np.random.random_integers(0,20000,samples_transfer)
#    original=[np.array([[agent._dataset._observations[o]._data[rand_ind[n]+l] for l in range(1)] for n in range(samples_transfer)]) for o in range(1)]
#    transfer=[np.array([[-agent._dataset._observations[o]._data[rand_ind[n]+l] for l in range(1)] for n in range(samples_transfer)]) for o in range(1)]
#
#    print ("original[0][0:10], transfer[0][0:10]")
#    print (original[0][0:10], transfer[0][0:10])
#
#    # Transfer between the two repr
#    #learning_algo.transfer(original, transfer, 5000000/samples_transfer)
#
#    
#    # --- Re instantiate environment with reverse=True ---
#    env = maze_env(rng, higher_dim_obs=HIGHER_DIM_OBS, reverse=True)
#
#    # --- Re instantiate agent ---
#    agent = NeuralAgent(
#        env,
#        learning_algo,
#        parameters.replay_memory_size,
#        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
#        parameters.batch_size,
#        rng,
#        test_policy=test_policy)
#
#    # --- Bind controllers to the agent ---
#    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and 
#    # learning rate as well as the training epoch number.
#    agent.attach(bc.VerboseController(
#        evaluate_on='epoch', 
#        periodicity=1))
#        
#    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we 
#    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
#    agent.attach(bc.LearningRateController(
#        initial_learning_rate=parameters.learning_rate, 
#        learning_rate_decay=parameters.learning_rate_decay,
#        periodicity=1))
#    
#    # Same for the discount factor.
#    agent.attach(bc.DiscountFactorController(
#        initial_discount_factor=parameters.discount, 
#        discount_factor_growth=parameters.discount_inc, 
#        discount_factor_max=parameters.discount_max,
#        periodicity=1))
#
#    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
#    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
#    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
#    # episode or epoch (or never, hence the resetEvery='none').
#    agent.attach(bc.EpsilonController(
#        initial_e=parameters.epsilon_start, 
#        e_decays=parameters.epsilon_decay, 
#        e_min=parameters.epsilon_min,
#        evaluate_on='action',
#        periodicity=1,
#        reset_every='none'))
#
#    agent.run(1, N_SAMPLES)
#    #print (agent._dataset._rewards._data[0:500])
#    #print (agent._dataset._terminals._data[0:500])
#    print("end gathering data")
#    # Setting the dataset to be the same than the old one (but modif for the observations)
#    agent._dataset._rewards._data=old_rewards
#    agent._dataset._terminals._data=old_terminals
#    agent._dataset._actions._data=old_actions
#    agent._dataset._observations[0]._data=-old_observations
#
#    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
#    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
#    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
#    agent.attach(bc.TrainerController(
#        evaluate_on='action', 
#        periodicity=parameters.update_frequency, 
#        show_episode_avg_V_value=True, 
#        show_avg_Bellman_residual=True))
#
#    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
#    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want 
#    # these validation epoch to interfere with the training of the agent, which is well established by the 
#    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole 
#    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the 
#    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards 
#    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every 
#    # [parameters.period_btw_summary_perfs] *validation* epochs.
#    agent.attach(bc.InterleavedTestEpochController(
#        id=maze_env.VALIDATION_MODE, 
#        epoch_length=parameters.steps_per_test,
#        controllers_to_disable=[0, 1, 2, 3, 4],
#        periodicity=2,
#        show_score=True,
#        summarize_every=1))
#
#
##    agent.attach(bc.InterleavedTestEpochController(
##        id=maze_env.VALIDATION_MODE+1, 
##        epoch_length=parameters.steps_per_test,
##        controllers_to_disable=[0, 1, 2, 3, 4, 5, 7,8],
##        periodicity=2,
##        show_score=True,
##        summarize_every=1))
##
##    agent.attach(bc.InterleavedTestEpochController(
##        id=maze_env.VALIDATION_MODE+2, 
##        epoch_length=parameters.steps_per_test,
##        controllers_to_disable=[0, 1, 2, 3, 4, 5, 6,8],
##        periodicity=2,
##        show_score=True,
##        summarize_every=1))
##    
##    agent.attach(bc.InterleavedTestEpochController(
##        id=maze_env.VALIDATION_MODE+3, 
##        epoch_length=parameters.steps_per_test,
##        controllers_to_disable=[0, 1, 2, 3, 4, 5, 6, 7],
##        periodicity=2,
##        show_score=True,
##        summarize_every=1))
#
#    agent.gathering_data=False
#    agent.run(parameters.epochs, parameters.steps_per_epoch)
#

