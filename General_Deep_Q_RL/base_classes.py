from theano import config
import numpy as np

class Environment(object):            
    def reset(self, mode):
        raise NotImplementedError()
        
    def act(self, action, mode):
        raise NotImplementedError()

    def batchDimensions(self):
        raise NotImplementedError()

    def nActions(self):
        raise NotImplementedError()

    def inTerminalState(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def summarizePerformance(self, mode, test_data_set):
        pass

class QNetwork(object):        
    def train(self, states, actions, rewards, nextStates, terminals):
        raise NotImplementedError()

    def chooseBestAction(self, state):
        raise NotImplementedError()

    def qValues(self, state):
        raise NotImplementedError()

    def setLearningRate(self, lr):
        raise NotImplementedError()

    def setDiscountFactor(self, df):
        raise NotImplementedError()

    def learningRate(self):
        raise NotImplementedError()

    def discountFactor(self):
        raise NotImplementedError()

if __name__ == "__main__":
    pass
