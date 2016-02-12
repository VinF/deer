"""
The NeuralAgent class wraps a deep Q-network for training and testing
in any given environment.

Modifications: Vincent Francois-Lavet
Contributor: David Taralla
"""

from theano import config
import os
import cPickle
import time
import logging
import numpy as np
import copy
import sys
import experiment.base_controllers as controllers
import utils as ut
from warnings import warn
from IPython import embed

class NeuralAgent(object):
    def __init__(self, environment, q_network, replay_memory_size, replay_start_size, batch_size, frameSkip, randomState):
        if replay_start_size < max(environment.batchDimensions()[0]):
            raise AgentError("Replay_start_size should be greater than the biggest history of a state.")

        self._controllers = []
        self._environment = environment
        self._network = q_network
        self._epsilon = 1
        self._replayMemorySize = replay_memory_size
        self._replayMemoryStartSize = replay_start_size
        self._batchSize = batch_size
        self._frameSkip = frameSkip
        self._randomState = randomState
        self._dataSet = DataSet(environment.batchDimensions(), maxSize=replay_memory_size, randomState=randomState)
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

    def epsilon(self):
        return self._epsilon

    def setLearningRate(self, lr):
        self._network.setLearningRate(lr)

    def learningRate(self):
        return self._network.learningRate()

    def setDiscountFactor(self, df):
        self._network.setDiscountFactor(df)

    def discountFactor(self):
        return self._network.discountFactor()

    def avgBellmanResidual(self):
        if (len(self._trainingLossAverages) == 0):
            return -1
        return np.average(self._trainingLossAverages)

    def avgEpisodeVValue(self):
        if (len(self._VsOnLastEpisode) == 0):
            return -1
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
        self._dataSetTest = DataSet(self._environment.batchDimensions(), self._randomState, maxSize=self._replayMemorySize)

    def endTesting(self):
        if not self._inTestingMode:
            warn("Testing mode was not ON.", AgentWarning)
        self._inTestingMode = False

    def summarizeTestPerformance(self):
        if not self._inTestingMode:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._dataSetTest)

    def train(self):
        if self._dataSet.nElems() < self._replayMemoryStartSize:
            return

        try:
            states, actions, rewards, next_states, terminals = self._dataSet.randomBatch(self._batchSize)
            loss = self._network.train(states, actions, rewards, next_states, terminals)
            self._trainingLossAverages.append(loss)
        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

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

            obs = self._environment.observe()
            V, action, reward = self._step()
            self._VsOnLastEpisode.append(V)
            isTerminal = self._environment.inTerminalState()
            if self._inTestingMode:
                self._totalTestReward += reward

            self._addSample(obs, action, reward, isTerminal)
            for c in self._controllers: c.OnActionTaken(self)
            
            if isTerminal:
                break
            
        self._inEpisode = False
        for c in self._controllers: c.OnEpisodeEnd(self, isTerminal, reward)
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
            reward += self._environment.act(action, self._inTestingMode)
            if self._environment.inTerminalState():
                break

        return V, action, reward

    def _addSample(self, ponctualObs, action, reward, isTerminal):
        if self._inTestingMode:
            self._dataSetTest.addSample(ponctualObs, action, reward, isTerminal)
        else:
            self._dataSet.addSample(ponctualObs, action, reward, isTerminal)


    def _chooseAction(self):
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
            action = self._network.chooseBestAction(state)
            V = max(self._network.qValues(state))
        else:
            if self._dataSet.nElems() > self._replayMemoryStartSize:
                # e-Greedy policy
                if self._randomState.rand() < self._epsilon:
                    action = self._randomState.randint(0, self._environment.nActions())
                    V = 0
                else:
                    action = self._network.chooseBestAction(state)
                    V = max(self._network.qValues(state))
            else:
                # Still gathering initial data: choose dummy action
                action = self._randomState.randint(0, self._environment.nActions())
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

class DataSet(object):
    """A replay memory consisting of circular buffers for observations, actions, and rewards.

    """
    def __init__(self, history_sizes, randomState=None, maxSize=1000):
        """Construct a DataSet.

        Arguments:
            history_sizes - For each input i, history_sizes[i] is the size of the history for this input.
            randomState - Numpy random number generator. If None, a new one is created with default numpy seed.
            maxSize - The maximum size of this buffer.

        """
        self._batchDimensions = history_sizes
        self._maxHistorySize = np.max([history_sizes[i][0] for i in range (len(history_sizes))])
        self._size = maxSize
        self._observations = np.zeros(len(history_sizes), dtype='object') # One list per input; will be initialized at 
        self._actions      = np.zeros(maxSize, dtype='int32')             # first call of addState
        self._rewards      = np.zeros(maxSize, dtype=config.floatX)
        self._terminals    = np.zeros(maxSize, dtype='bool')

        if (randomState == None):
            self._randomState = np.random.RandomState()
        else:
            self._randomState = randomState

        self._nElems  = 0

    def slice(self, fromIndex, toIndex):
        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input][fromIndex:toIndex]

        return ret

    def randomBatch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and next_states for batch_size randomly 
        chosen state transitions. Note that if terminal[i] == True, then 
        next_states[input][i] == np.zeros_like(states[input][i]) for all 'input's.
        
        Arguments:
            batch_size - Number of elements in the batch.

        Returns:
            states - An ndarray(size=number_of_inputs, dtype='object), where states[input] is a 2+D matrix of dimensions
                     batch_size x input.historySize x "shape of a given ponctual observation for this input". States were
                     taken randomly in the data set such that they are complete regarding the histories of each input.
            actions - The actions taken in each of those states.
            rewards - The rewards obtained for taking these actions in those states.
            next_states - Same structure than states, but next_states[i][j] is guaranteed to be the information 
                          concerning the state following the one described by states[i][j] for input i.
            terminals - Whether these actions lead to terminal states.

        Throws:
            SliceError - If a batch of this size could not be built based on current data set (not enough data or 
                         all trajectories are too short).
        """

        rndValidIndices = np.zeros(batch_size, dtype='int32')

        for i in range(batch_size): # TODO: multithread this loop?
            rndValidIndices[i] = self._randomValidStateIndex()
            
        
        actions   = self._actions[rndValidIndices]
        rewards   = self._rewards[rndValidIndices]
        terminals = self._terminals[rndValidIndices]
        states = np.zeros(len(self._batchDimensions), dtype='object')
        next_states = np.zeros_like(states)
        
        for input in range(len(self._batchDimensions)):
            states[input] = np.zeros((batch_size,) + self._batchDimensions[input], dtype=self._observations[input].dtype)
            next_states[input] = np.zeros_like(states[input])
            for i in range(batch_size):
                states[input][i] = self._observations[input][rndValidIndices[i]+1-self._batchDimensions[input][0]:rndValidIndices[i]+1]
                if rndValidIndices[i] >= self._size - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    next_states[input][i] = self._observations[input][rndValidIndices[i]+2-self._batchDimensions[input][0]:rndValidIndices[i]+2]

        return states, actions, rewards, next_states, terminals

    def _randomValidStateIndex(self):
        lowerBound = self._size - self._nElems
        index_lowerBound = lowerBound + self._maxHistorySize - 1
        try:
            index = self._randomState.randint(index_lowerBound, self._size)
        except ValueError:
            raise SliceError("There aren't enough elements in the dataset to create a complete state ({} elements "
                             "in dataset; requires {}".format(self._nElems, self._maxHistorySize))

        # Check if slice is valid wrt terminals
        firstTry = index
        startWrapped = False
        while True:
            i = index-1
            processed = 0
            for _ in range(self._maxHistorySize-1):
                if (i < lowerBound or self._terminals[i]):
                    break;

                i -= 1
                processed += 1
            
            if (processed < self._maxHistorySize - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < index_lowerBound):
                    startWrapped = True
                    index = self._size - 1
                if (startWrapped and index <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index
       

    def nElems(self):
        """Return the number of *complete* samples in this data set (i.e. complete tuples (state, action, reward, isTerminal)).
        Might thus be different than what nStates returns.

        """
        return self._nElems


    def addSample(self, ponctualObs, action, reward, isTerminal):
        """Store a (state, action, reward, isTerminal) in the dataset. 
        Equivalent to 'addState(state); addActionRewardTerminal(action, reward, isTerminal)'.

        Arguments:
            ponctualObs - An ndarray(dtype='object') whose length is the number of inputs.
                          For each input i, observation[i] is a 2D matrix that represents actual data.
            action - The id of the action taken in the last inserted state using addState.
            reward - The reward associated to taking 'action' in the last inserted state using addState.
            isTerminal - Tells whether 'action' lead to a terminal state (i.e. whether the tuple (state, action, reward, isTerminal) marks the end of a trajectory).

        """
        # Initialize the observations container if necessary
        if (self._nElems == 0):
            for i in range(len(ponctualObs)):
                self._observations[i] = np.zeros((self._size,) + np.array(ponctualObs[i]).shape)
        
        # Store observations
        for i in range(len(self._batchDimensions)):
            ut.appendCircular(self._observations[i], ponctualObs[i])
        
        # Store rest of sample
        ut.appendCircular(self._actions, action)
        ut.appendCircular(self._rewards, reward)
        ut.appendCircular(self._terminals, isTerminal)

        if (self._nElems < self._size):
            self._nElems += 1

        
class SliceError(LookupError):
    """Exception raised for errors when getting slices from CircularBuffers.

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
