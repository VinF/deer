"""Toy environment launcher. See the docs for more details about this environment.

"""

import numpy as np

from deer.agent import NeuralAgent
from deer.learning_algos.q_net_keras import MyQNetwork
from Toy_env import MyEnv as Toy_env
import deer.experiment.base_controllers as bc


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

# We also want to interleave a "test epoch" between each training epoch. 
agent.attach(bc.InterleavedTestEpochController(epoch_length=500))
    
# --- Run the experiment ---
agent.run(n_epochs=100, epoch_length=1000)
