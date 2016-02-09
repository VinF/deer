class Environment(object):
    def __init__(self, rng):
        raise NotImplementedError()
                
    def reset(self, testing):
        raise NotImplementedError()
        
    def act(self, action, testing):
        raise NotImplementedError()

    def batchDimensions(self):
        raise NotImplementedError()

    def inTerminalState(self):
        raise NotImplementedError()

    def isSuccess(self):
        raise NotImplementedError()

    def observe(self):
        raise NotImplementedError()

    def state(self):
        raise NotImplementedError()

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
