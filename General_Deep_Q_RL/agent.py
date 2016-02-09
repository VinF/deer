"""
The NeuralAgent class wraps a deep Q-network for training and testing
in any given environment.

Modifications: Vincent Francois-Lavet
Contributor: David Taralla
"""

import os
import cPickle
import time
import logging
import numpy as np
import data_set
import copy
import sys
import experiment.base_controllers as controllers
from warnings import warn
sys.setrecursionlimit(10000)


class NeuralAgent(object):
    def __init__(self, environment, q_network, replay_memory_size, replay_start_size, batch_size, frameSkip, randomState):
        self._controllers = []
        self._environment = environment
        self._network = q_network
        self._epsilon = 1
        self._replayMemorySize = replay_memory_size
        self._replayMemoryStartSize = replay_start_size
        self._batchSize = batch_size
        self._frameSkip = frameSkip
        self._randomState = randomState
        self._dataSet = data_set.DataSet(environment.batchDimensions(), maxSize=replay_memory_size, randomState=randomState)
        self._dataSetTest = None # Will be created by startTesting() when necessary
        self._inTestingMode = False
        self._testEpochsLength = 0
        self._totalTestReward = 0
        self._trainingLossAverages = []
        self._VsOnLastEpisode = []
        self._inEpisode = False

    def setControllersActive(self, toDisable, active):
        for i in toDisable:
            self._controllers[i].setActive(active)

    def setEpsilon(self, e):
        self._epsilon = e

    def setLearningRate(self, lr):
        self._network.setLearningRate(lr)

    def setDiscountFactor(self, df):
        self._network.setDiscountFactor(df)

    def avgBellmanResidual(self):
        return np.average(self._trainingLossAverages)

    def avgEpisodeVValue(self):
        return np.average(self._VsOnLastEpisode)
    
    def totalRewardOverLastTest(self):
        return self._totalTestReward

    def attach(self, controller):
        if (isinstance(controller, controllers.Controller)):
            self._controllers.append(controller)
        else:
            raise TypeError("The object you try to attach is not a Controller.")

    def detach(self, controllerIdx):
        return self._controllers.pop(controllerIdx)

    def startTesting(self, epochLength):
        if self._inTestingMode:
            warn("Testing mode is already ON.", AgentWarning)
        if self._inEpisode:
            raise AgentError("Trying to start testing while current episode is not yet finished. This method can be "
                             "called only *between* episodes.")

        self._inTestingMode = True
        self._testEpochsLength = epochLength
        self._totalTestReward = 0
        self._dataSetTest = data_set.DataSet(self._environment.batchDimensions(), self._randomState, maxSize=self._replayMemorySize)

    def endTesting(self):
        if not self._inTestingMode:
            warn("Testing mode was not ON.", AgentWarning)
        self._inTestingMode = False

    def summarizeTestPerformance(self):
        if not self._inTestingMode:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._dataSetTest)

    def train(self):
        try:
            states, actions, rewards, next_states, terminals = self._dataSet.randomBatch(self._batchSize)
            loss = self._network.train(states, actions, rewards, next_states, terminals)
            self._trainingLossAverages.append(loss)
        except data_set.SliceError as e:
            warn("Training not done - " + e, AgentWarning)

    def run(self, nEpochs, epochLength):
        for c in self._controllers: c.OnStart(self)
        for _ in range(nEpochs):
            if self._inTestingMode:
                while self._testEpochsLength > 0:
                    self._testEpochsLength = self._runEpisode(self._testEpochsLength)
            else:
                length = epochLength
                while length > 0:
                    length = self._runEpisode(length)

            for c in self._controllers: c.OnEpochEnd(self)
            


    def _runEpisode(self, maxSteps):
        self._inEpisode = True
        self._environment.reset(self._inTestingMode)
        
        self._trainingLossAverages = []
        self._VsOnLastEpisode = []
        while maxSteps > 0:
            maxSteps -= 1

            V, action, reward = self._step()
            self._VsOnLastEpisode.append(V)
            isTerminal = self._environment.inTerminalState()
            if self._inTestingMode:
                self._totalTestReward += reward

            self._addSample(self._environment.observe(), action, reward, isTerminal)
            for c in self._controllers: c.OnActionTaken(self)
            
            if isTerminal:
                break
            
        self._inEpisode = False
        for c in self._controllers: c.OnEpisodeEnd(self, isTerminal, self._environment.isSuccess())
        return maxSteps

        
    def _step(self):
        """
        This method is called at each time step. If the agent is currently in testing mode, and if its *test* replay 
        memory has enough samples, it will select the best action it can. If there are not enough samples, FIXME.
        In the case the agent is not in testing mode, if its replay memory has enough samples, it will select the best 
        action it can with probability 1-CurrentEpsilon and a random action otherwise. If there are not enough samples, 
        it will always select a random action.

        Arguments:
           state - An ndarray(size=number_of_inputs, dtype='object), where states[input] is a 1+D matrix of dimensions
                   input.historySize x "shape of a given ponctual observation for this input".

        Returns:
           action - The id of the action selected by the agent.
           V - Estimated value function of current state.
        """
        
        action, V = self._chooseAction()        
        reward = 0
        for _ in range(self._frameSkip):
            reward += self._environment.act(action)
            if self._environment.inTerminalState():
                break

        return V, action, reward

    def _addSample(self, ponctualObs, action, reward, isTerminal):
        if self._inTestingMode:
            self._dataSetTest.addSample(ponctualObs, action, reward, isTerminal)
        else:
            self._dataSet.addSample(ponctualObs, action, reward, isTerminal)


    def _chooseAction(self, epsilon):
        """
        Get the action chosen by the agent regarding epsilon greedy parameter and current state. It will be a random
        action with probability epsilon and the believed-best action otherwise.

        Arguments:
           epsilon - float, exploration of the epsilon greedy
           state - An ndarray(size=number_of_inputs, dtype='object), where states[input] is a 1+D matrix of dimensions
                   input.historySize x "shape of a given ponctual observation for this input".

        Returns:
           action - The id of the chosen action
           An integer - action based on the current policy
        """
        
        state = self._environment.state()
        if self._inTestingMode:
            if self._dataSetTest.nElems() > self._replayMemoryStartSize:
                # Use gathered data to choose action
                action = self._network.chooseBestAction(state)
                V = max(self._network.qValues(state))
            else:
                # Still gathering initial data: choose dummy action
                action = 0 #self.rng.randint(0, self.num_actions) #TODO: ask vincent if =0 is not a bug?
                V = 0
        else:
            if self._dataSet.nElems() > self._replayMemoryStartSize:
                # e-Greedy policy
                if self._randomState.rand() < epsilon:
                    action = self._randomState.randint(0, self._environment.num_actions)
                    V = 0
                else:
                    action = self._network.chooseBestAction(state)
                    V = max(self._network.qValues(state))
            else:
                # Still gathering initial data: choose dummy action
                action = self._randomState.randint(0, self._environment.num_actions)
                V = 0
                
        for c in self._controllers: c.OnActionChosen(self, action)
        return action, V

 
class AgentError(RuntimeError):
    """Exception raised for errors when calling the various Agent methods at wrong times.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class AgentWarning(RuntimeWarning):
    """Warning issued of the various Agent methods.

    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

if __name__ == "__main__":
    pass
