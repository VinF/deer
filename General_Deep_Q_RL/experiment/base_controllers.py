"""This class should be the base class of any controller you would want to define. 

Author: Vincent Francois-Lavet, David Taralla
"""

class Controller(object):
    """A base controller that does nothing when receiving the various signals emitted by an agent.
    """

    def __init__(self):
        self._active = True

    def setActive(self, active):
        self._active = active

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
        super(Controller, self).__init__()
        self._epochCount = 0
        self._initLr = initialLearningRate
        self._lr = initialLearningRate
        self._lrDecay = learningRateDecay
        self._periodicity = periodicity

    def OnStart(self, agent):
        if (self._active == False):
            return

        self._epochCount = 0
        agent.setLearningRate(self._initLr)
        self._lr = self._initLr * self._lrDecay

    def OnEpochEnd(self, agent):
        if (self._active == False):
            return

        self._epochCount += 1
        if self._periodicity <= 1 or self._epochCount % self._periodicity == 0:
            agent.setLearningRate(self._lr)
            self._lr *= self._lrDecay


class EpsilonController(Controller):
    """A controller that modifies the probability of taking a random action periodically.

    Arguments:
        initialE - Start epsilon
        eDecays - How many steps of "evaluateOn" are necessary for epsilon to reach eMin
        eMin - End epsilon
        evaluateOn - After what type of event epsilon shoud be updated periodically ('action', 'episode', 'epoch').
        periodicity - How many steps of "evaluateOn" are necessary before an update of epsilon
        resetEvery - After what type of event epsilon should be reset to initial value ('none', 'episode', 'epoch').
    """
    def __init__(self, initialE, eDecays, eMin, evaluateOn='action', periodicity=1, resetEvery='none'):
        super(Controller, self).__init__()
        self._count = 0
        self._initE = initialE
        self._e = initialE
        self._eMin = eMin
        self._eDecay = (initialE - eMin) / eDecays
        self._periodicity = periodicity

        self._onAction = 'action' == evaluateOn
        self._onEpisode = 'episode' == evaluateOn
        self._onEpoch = 'epoch' == evaluateOn
        if not self._onAction and not self._onEpisode and not self._onEpoch:
            self._onAction = True

        self._resetOnEpisode = 'episode' == resetEvery
        self._resetOnEpoch = 'epoch' == resetEvery

    def OnStart(self, agent):
        if (self._active == False):
            return

        _reset(agent)

    def OnEpisodeEnd(self, agent, terminalReached, successful):
        if (self._active == False):
            return

        if _resetOnEpisode:
            _reset(agent)
        elif _onEpisode:
            _update(agent)

    def OnEpochEnd(self, agent):
        if (self._active == False):
            return

        if _resetOnEpoch:
            _reset(agent)
        elif _onEpoch:
            _update(agent)

    def OnActionChosen(self, agent, action):
        if (self._active == False):
            return

        if _onAction:
            _update(agent)


    def _reset(self, agent):
        self._count = 0
        agent.setEpsilon(self._initE)
        self._e = max(self._e - self._eDecay, self._eMin)

    def _update(self, agent):
        self._count += 1
        if self._periodicity <= 1 or self._count % self._periodicity == 0:
            agent.setEpsilon(self._e)
            self._e = max(self._e - self._eDecay, self._eMin)


class DiscountFactorController(Controller):
    """A controller that modifies the qnetwork discount periodically.

    """
    def __init__(self, initialDiscountFactor, discountFactorGrowth, periodicity=1):
        super(Controller, self).__init__()
        self._epochCount = 0
        self._initDF = initialDiscountFactor
        self._df = initialDiscountFactor
        self._dfGrowth = discountFactorGrowth
        self._periodicity = periodicity

    def OnStart(self, agent):
        if (self._active == False):
            return

        self._epochCount = 0
        agent.setDiscountFactor(self._initDF)
        if (self._initDF < 0.99):
            self._df = 1 - (1 - self._initDF) * self._dfGrowth
        else:
            self._df = self._initDF

    def OnEpochEnd(self, agent):
        if (self._active == False):
            return

        self._epochCount += 1
        if self._periodicity <= 1 or self._epochCount % self._periodicity == 0:
            if (self._df < 0.99):
                agent.setDiscountFactor(self._df)
                self._df = 1 - (1 - self._df) * self._dfGrowth


class InterleavedTestEpochController(Controller):
    """A controller that interleaves a test epoch between training epochs of the agent.

    """
    def __init__(self, epochLength, controllersToDisable=[], periodicity=2):
        super(Controller, self).__init__()
        self._epochCount = 0
        self._epochLength = epochLength
        self._toDisable = controllersToDisable
        if periodicity <= 2:
            self._periodicity = 2
        else:
            self._periodicity = periodicity

    def OnStart(self, agent):
        if (self._active == False):
            return

        self._epochCount = 0

    def OnEpochEnd(self, agent):
        if (self._active == False):
            return

        mod = self._epochCount % self._periodicity
        self._epochCount += 1
        if mod == 0:
            agent.startTesting(self._epochLength)
            agent.setControllersActive(self._toDisable, False)
        elif mod == 1:
            agent.endTesting()
            agent.setControllersActive(self._toDisable, True)



if __name__ == "__main__":
    pass
