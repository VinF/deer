"""Toy environment launcher. See the docs for more details about this environment.

Authors: Vincent Francois-Lavet, David Taralla
"""

import numpy as np

from deer.agent import NeuralAgent
from deer.q_networks.q_net_theano import MyQNetwork
from Toy_env import MyEnv as Toy_env
import deer.experiment.base_controllers as bc

if __name__ == "__main__":

    rng = np.random.RandomState(123456)
    
    # --- Instantiate environment ---
    env = Toy_env(rng)

    # --- Instantiate qnetwork ---
    qnetwork = MyQNetwork(
        environment=env,
        random_state=rng)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        random_state=rng)

    # --- Bind controllers to the agent ---
    # Before every training epoch, we want to print a summary of the agent's epsilon, discount and 
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController())

    # During training epochs, we want to train the agent after every action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode.
    agent.attach(bc.TrainerController())

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a 
    # "test epoch" between each training epoch. We do not want these test epoch to interfere with the training of the 
    # agent. Therefore, we will disable these controllers for the whole duration of the test epochs interleaved this 
    # way, using the controllersToDisable argument of the InterleavedTestEpochController. The value of this argument 
    # is a list of the indexes of all controllers to disable, their index reflecting in which order they were added.
    agent.attach(bc.InterleavedTestEpochController(
        epoch_length=500, 
        controllers_to_disable=[0, 1]))
        
    # --- Run the experiment ---
    agent.run(n_epochs=100, epoch_length=1000)
