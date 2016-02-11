from theano import config
import numpy as np

class Environment(object):                
    def _initState(self):
        self._state = []
        batchDims = self.batchDimensions()
        for i in range(len(batchDims)):
            self._state.append(np.zeros(batchDims[i], dtype=config.floatX))
    
    def _updateState(self):
        obs = self.observe()
        for i in range(len(obs)):
            if (self._state[i].ndim == 2):
                self._state[i] = np.roll(self._state[i], -1, axis=0)
            else:
                self._state[i] = np.roll(self._state[i], -1)
            self._state[i][-1] = obs[i]
            
    def reset(self, testing):
        raise NotImplementedError()
        
    def act(self, action, testing):
        raise NotImplementedError()

    def batchDimensions(self):
        raise NotImplementedError()

    def nActions(self):
        raise NotImplementedError()

    def inTerminalState(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def state(self):
        return self._state

    def summarizePerformance(self, test_data_set):
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
