"""This file defines proxies to make it easy to test parts of the API.

Author: David Taralla
"""

from base_classes import QNetwork

class MyQNetwork(QNetwork):
    def __init__(self, environment, batchSize):
        self._env = environment
        self._batchSize = batchSize
        self._trainPassed = False

    def train(self, states, actions, rewards, nextStates, terminals):
        if self._trainPassed:
            return 0.42

        # Let's check arguments are correct
        self._checkStates(states)
        self._checkStates(nextStates)

        # action, terminals and rewards: should have _batchSize of each of them
        if len(actions) != self._batchSize:
            raise UnitTestError("{} != {}".format(len(actions), self._batchSize))
        if len(rewards) != self._batchSize:
            raise UnitTestError("{} != {}".format(len(rewards), self._batchSize))
        if len(terminals) != self._batchSize:
            raise UnitTestError("{} != {}".format(len(terminals), self._batchSize))

        self._trainPassed = True

        # Return dummy loss to make caller happy
        return 0.42

    def chooseBestAction(self, state):
        _checkState(state)
        return 0

    def qValues(self, state):
        _checkState(state)
        return [0]

    def setLearningRate(self, lr):
        pass

    def setDiscountFactor(self, df):
        pass

    def learningRate(self):
        return 0

    def discountFactor(self):
        return 0

    def _checkStates(self, states):
        # states should be nObservations * batchSize * env.batchDimensions[i]
        dims = self._env.batchDimensions()

        if len(states) != len(dims):
            raise UnitTestError("{} != {}".format(len(states), len(dims)))

        for i in range(len(states)):
            if states[i].shape != (self._batchSize,) + dims[i]:
                raise UnitTestError("Observation {}: {} != {}".format(i, states[i].shape, (self._batchSize,) + dims[i]))

    def _checkState(self, state):
        # state should be nObservations * env.batchDimensions[i]
        dims = self._env.batchDimensions()

        if len(states) != len(dims):
            raise UnitTestError("{} != {}".format(len(states), len(dims)))

        for i in range(len(states)):
            if states[i].shape != dims[i]:
                raise UnitTestError("Observation {}: {} != {}".format(i, states[i].shape, dims[i]))



class UnitTestError(RuntimeError):
    """Exception raised when unit tests fail.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

if __name__ == "__main__":
    pass
