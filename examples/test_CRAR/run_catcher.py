""" Catcher launcher

"""

import sys
import logging
import numpy as np
from joblib import hash, dump
import os

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.CRAR_modif_keras import CRAR
from catcher_env import MyEnv as catcher_env
import deer.experiment.base_controllers as bc

from deer.policies import EpsilonGreedyPolicy


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 50
    STEPS_PER_TEST = 500
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
    LEARNING_RATE_DECAY = 0.9
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
HIGH_INT_DIM = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = catcher_env(rng, higher_dim_obs=HIGHER_DIM_OBS, reverse=False)
    
    # --- Instantiate learning algorithm ---
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
        internal_dim=3)
    
    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.1)#1.)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        learning_algo,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
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
        
    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")
    
    #agent.run(n_epochs=1, epoch_length=20000) #For collecting data off-policy
    #print "end gathering data"
    
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
        
    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want 
    # these validation epoch to interfere with the training of the agent, which is well established by the 
    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole 
    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the 
    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards 
    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every 
    # [parameters.period_btw_summary_perfs] *validation* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=catcher_env.VALIDATION_MODE, 
        epoch_length=parameters.steps_per_test,
        periodicity=1,
        show_score=True,
        summarize_every=1))

    #agent.gathering_data=False
    agent.run(parameters.epochs, parameters.steps_per_epoch)
    

#    ###
#    # TRANSFER
#    ###
#    optimized_params=learning_algo.getAllParams()
#    print ("The optimized_params are")
#    print (optimized_params)
#
#    # --- Instantiate learning_algo ---
#    learning_algo = CRAR(
#        env,
#        parameters.rms_decay,
#        parameters.rms_epsilon,
#        parameters.momentum,
#        parameters.clip_norm,
#        parameters.freeze_interval,
#        parameters.batch_size,
#        parameters.update_rule,
#        rng,
#        double_Q=True,
#        high_int_dim=HIGH_INT_DIM,
#        internal_dim=3)
#    learning_algo.setAllParams(optimized_params)
#
#    samples_transfer=500
#    rand_ind=np.random.random_integers(0,20000,samples_transfer)
#    original=[np.array([[agent._dataset._observations[o]._data[rand_ind[n]+l] for l in range(1)] for n in range(samples_transfer)]) for o in range(1)]
#    transfer=[np.array([[-agent._dataset._observations[o]._data[rand_ind[n]+l] for l in range(1)] for n in range(samples_transfer)]) for o in range(1)]
#
#    print ("original[0][0:10], transfer[0][0:10]")
#    print (original[0][0:10], transfer[0][0:10])
#
#    # Transfer between the two repr
#    learning_algo.transfer(original, transfer, 5000)
#
#    
#    # --- Instantiate environment with reverse=True ---
#    env = catcher_env(rng, higher_dim_obs=HIGHER_DIM_OBS, reverse=True)
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
#    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
#    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
#    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
#    agent.attach(bc.TrainerController(
#        evaluate_on='action', 
#        periodicity=parameters.update_frequency, 
#        show_episode_avg_V_value=True, 
#        show_avg_Bellman_residual=True))
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
#    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
#    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want 
#    # these validation epoch to interfere with the training of the agent, which is well established by the 
#    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole 
#    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the 
#    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards 
#    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every 
#    # [parameters.period_btw_summary_perfs] *validation* epochs.
#    agent.attach(bc.InterleavedTestEpochController(
#        id=catcher_env.VALIDATION_MODE, 
#        epoch_length=parameters.steps_per_test,
#        controllers_to_disable=[0, 1, 2, 3, 4],
#        periodicity=2,
#        show_score=True,
#        summarize_every=1))
#
#
#    #agent.gathering_data=False
#    agent.run(parameters.epochs, parameters.steps_per_epoch)
#
#
#    #print "agent.DataSet.self._terminals"
#    #print "agent._dataset.terminals()"
#    #print agent._dataset.terminals()
#    #print agent._dataset._terminals._data[0:2000]
#    #print agent._dataset._actions._data[0:2000]
##    r=agent._dataset._rewards._data[0:2000]
##    print "r before"
##    print r
#    print agent._dataset._observations[0]._data[0:10]
##    ind=np.argwhere(r>0)
##    print "agent._dataset._observations[0]._data[ind[0]]"
##    print agent._dataset._observations[0]._data[ind[0]]
##    print ind
##    agent._dataset._rewards._data=np.delete(agent._dataset._rewards._data,ind)
##    agent._dataset._terminals._data=np.delete(agent._dataset._terminals._data,ind)
##    agent._dataset._actions._data=np.delete(agent._dataset._actions._data,ind)
##    agent._dataset._observations[0]._data=np.delete(agent._dataset._observations[0]._data,ind,axis=0)
##    r=agent._dataset._rewards._data[0:2000]
##    print "r after"
##    print r
##    print "agent._dataset._observations[0]._data[ind[0]] after"
##    print agent._dataset._observations[0]._data[ind[0]]
##
#
#
#
#    
#    # --- Show results ---
#    basename = "scores/" + fname
#    scores = joblib.load(basename + "_scores.jldump")
#    plt.plot(range(1, len(scores['vs'])+1), scores['vs'], label="VS", color='b')
#    plt.legend()
#    plt.xlabel("Number of epochs")
#    plt.ylabel("Score")
#    plt.savefig(basename + "_scores.pdf")
#    plt.show()
