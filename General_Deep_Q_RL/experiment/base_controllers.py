"""This class should be the base class of any controller you would want to define. 

Author: Vincent Francois-Lavet, David Taralla
"""

class Controller(object):
    """A base controller that does nothing when receiving the various signals emitted by an agent.

    """
    def OnStart(self, agent):
        pass

    def OnEpisodeEnd(self, agent, terminalReached, successful):
        pass

    def OnEpochEnd(self, agent):
        pass

    def OnBeforeTraining(self, agent):
        pass

    def OnEndTraining(self, agent):
        pass

    def OnActionChosen(self, agent, action):
        pass


class LearningRateController(Controller):
    """A controller that modifies the learning rate periodically.

    """
    def __init__(self, initialLearningRate, learningRateDecay, periodicity=1):
        self._epochCount = 0
        self._initLr = initialLearningRate
        self._lr = initialLearningRate
        self._lrDecay = learningRateDecay
        self._periodicity = periodicity

    def OnStart(self, agent):
        self._epochCount = 0
        agent.SetLearningRate(self._initLr)
        self._lr = self._initLr * self._lrDecay

    def OnEpochEnd(self, agent):
        self._epochCount += 1
        if self._periodicity <= 1 or self._epochCount % self._periodicity == 0:
            agent.SetLearningRate(self._lr)
            self._lr *= self._lrDecay


class DiscountFactorController(Controller):
    """A controller that modifies the qnetwork discount periodically.

    """
    def __init__(self, initialDiscountFactor, discountFactorGrowth, periodicity=1):
        self._epochCount = 0
        self._initDF = initialDiscountFactor
        self._df = initialDiscountFactor
        self._dfGrowth = discountFactorGrowth
        self._periodicity = periodicity

    def OnStart(self, agent):
        self._epochCount = 0
        agent.SetDiscountFactor(self._initDF)
        if (self._initDF < 0.99):
            self._df = 1 - (1 - self._initDF) * self._dfGrowth
        else:
            self._df = self._initDF

    def OnEpochEnd(self, agent):
        self._epochCount += 1
        if self._periodicity <= 1 or self._epochCount % self._periodicity == 0:
            if (self._df < 0.99):
                agent.SetDiscountFactor(self._df)
                self._df = 1 - (1 - self._df) * self._dfGrowth


if __name__ == "__main__":
    pass
