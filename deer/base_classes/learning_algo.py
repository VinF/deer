"""
This module defines the base class for the learning algorithms.

"""

import numpy as np

class LearningAlgo(object):
    """ All the Q-networks, actor-critic networks, etc. should inherit this interface.

    Parameters
    -----------
    environment : object from class Environment
        The environment linked to the Q-network
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    """
    def __init__(self, environment, batch_size):
        self._environment = environment
        self._df = 0.9
        self._lr = 0.005
        self._input_dimensions = self._environment.inputDimensions()
        self._n_actions = self._environment.nActions()
        self._batch_size = batch_size

    def train(self, states, actions, rewards, nextStates, terminals):
        """ This method performs the training step (e.g. using Bellman iteration in a deep Q-network) 
        for one batch of tuples.
        """
        raise NotImplementedError()

    def chooseBestAction(self, state):
        """ Get the best action for a pseudo-state
        """        
        raise NotImplementedError()

    def qValues(self, state):
        """ Get the q value for one pseudo-state
        """        
        raise NotImplementedError()

    def setLearningRate(self, lr):
        """ Setting the learning rate
        NB: The learning rate has usually to be set in the optimizer, hence this function should
        be overridden. Otherwise, the learning rate change is likely not to be taken into account

        Parameters
        -----------
        lr : float
            The learning rate that has to bet set
        """
        self._lr = lr

    def setDiscountFactor(self, df):
        """ Setting the discount factor

        Parameters
        -----------
        df : float
            The discount factor that has to bet set
        """
        if df < 0. or df > 1.:
            raise AgentError("The discount factor should be in [0,1]")

        self._df = df

    def learningRate(self):
        """ Getting the learning rate
        """
        return self._lr

    def discountFactor(self):
        """ Getting the discount factor
        """
        return self._df

if __name__ == "__main__":
    pass
