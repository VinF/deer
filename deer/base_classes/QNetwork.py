"""
.. Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import numpy as np

class QNetwork(object):
    """ All the Q-networks classes should inherit this interface.

    Parameters
    -----------
    environment : object from class Environment
        The environment linked to the Q-network
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    """
    def __init__(self, environment, batchSize):
        self._environment = environment
        self._df = 0
        self._lr = 0
        self._inputDimensions = self._environment.inputDimensions()
        self._nActions = self._environment.nActions()
        self._batchSize = batchSize

    def train(self, states, actions, rewards, nextStates, terminals):
        """ This method performs the Bellman iteration for one batch of tuples.
        """
        raise NotImplementedError()

    def chooseBestAction(self, state):
        """ Get the best action for a belief state
        """        
        raise NotImplementedError()

    def qValues(self, state):
        """ Get the q value for one belief state
        """        
        raise NotImplementedError()

    def setLearningRate(self, lr):
        """ Setting the learning rate

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
