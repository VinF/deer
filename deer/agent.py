"""This module contains classes used to define any agent wrapping a DQN.

Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import os
import numpy as np
import copy
import sys
import joblib
from .experiment import base_controllers as controllers
from warnings import warn

class NeuralAgent(object):
    """The NeuralAgent class wraps a deep Q-network for training and testing in a given environment.
    
    Attach controllers to it in order to conduct an experiment (when to train the agent, when to test,...).
    """

    def __init__(self, environment, q_network, replay_memory_size, replay_start_size, batch_size, randomState):
        inputDims = environment.inputDimensions()

        if replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))):
            raise AgentError("Replay_start_size should be greater than the biggest history of a state.")
        
        self._controllers = []
        self._environment = environment
        self._network = q_network
        self._epsilon = 1
        self._replayMemorySize = replay_memory_size
        self._replayMemoryStartSize = replay_start_size
        self._batchSize = batch_size
        self._randomState = randomState
        self._dataSet = DataSet(environment, maxSize=replay_memory_size, randomState=randomState)
        self._tmpDataSet = None # Will be created by startTesting() when necessary
        self._mode = -1
        self._modeEpochsLength = 0
        self._totalModeReward = 0
        self._trainingLossAverages = []
        self._VsOnLastEpisode = []
        self._inEpisode = False
        self._selectedAction = -1
        self._state = []
        for i in range(len(inputDims)):
            self._state.append(np.zeros(inputDims[i], dtype=config.floatX))

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

    def overrideNextAction(self, action):
        self._selectedAction = action

    def avgBellmanResidual(self):
        if (len(self._trainingLossAverages) == 0):
            return -1
        return np.average(self._trainingLossAverages)

    def avgEpisodeVValue(self):
        if (len(self._VsOnLastEpisode) == 0):
            return -1
        return np.average(self._VsOnLastEpisode)

    def totalRewardOverLastTest(self):
        return self._totalModeReward

    def bestAction(self):
        action = self._network.chooseBestAction(self._state)
        V = max(self._network.qValues(self._state))
        return action, V
     
    def attach(self, controller):
        if (isinstance(controller, controllers.Controller)):
            self._controllers.append(controller)
        else:
            raise TypeError("The object you try to attach is not a Controller.")

    def detach(self, controllerIdx):
        return self._controllers.pop(controllerIdx)

    def mode(self):
        return self._mode

    def startMode(self, mode, epochLength):
        if self._inEpisode:
            raise AgentError("Trying to start mode while current episode is not yet finished. This method can be "
                             "called only *between* episodes for testing and validation.")
        elif mode == -1:
            raise AgentError("Mode -1 is reserved and means 'training mode'; use resumeTrainingMode() instead.")
        else:
            self._mode = mode
            self._modeEpochsLength = epochLength
            self._totalModeReward = 0
            del self._tmpDataSet
            self._tmpDataSet = DataSet(self._environment, self._randomState, maxSize=self._replayMemorySize)

    def resumeTrainingMode(self):
        self._mode = -1

    def summarizeTestPerformance(self):
        if self._mode == -1:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._tmpDataSet)

    def train(self):
        if self._dataSet.nElems() < self._replayMemoryStartSize:
            return

        try:
            states, actions, rewards, next_states, terminals = self._dataSet.randomBatch(self._batchSize)
            loss = self._network.train(states, actions, rewards, next_states, terminals)
            self._trainingLossAverages.append(loss)
        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

    def dumpNetwork(self, fname, nEpoch):
        try:
            os.mkdir("nnets")
        except Exception:
            pass
        basename = "nnets/" + fname

        for f in os.listdir("nnets/"):
            if fname in f:
                os.remove("nnets/" + f)

        all_params, all_params_conv = self._network.toDump()
        joblib.dump([all_params, all_params_conv], basename + ".epoch={}".format(nEpoch))
                

    def run(self, nEpochs, epochLength):
        for c in self._controllers: c.OnStart(self)
        i = 0
        while i < nEpochs or self._modeEpochsLength > 0:
            if self._mode != -1:
                while self._modeEpochsLength > 0:
                    self._modeEpochsLength = self._runEpisode(self._modeEpochsLength)
            else:
                length = epochLength
                while length > 0:
                    length = self._runEpisode(length)
                i += 1
            for c in self._controllers: c.OnEpochEnd(self)
            
        for c in self._controllers: c.OnEnd(self)

    def _runEpisode(self, maxSteps):
        self._inEpisode = True
        initState = self._environment.reset(self._mode)
        inputDims = self._environment.inputDimensions()
        for i in range(len(inputDims)):
            if inputDims[i][0] > 1:
                self._state[i][1:] = initState[i][1:]
        
        self._trainingLossAverages = []
        self._VsOnLastEpisode = []
        while maxSteps > 0:
            maxSteps -= 1

            obs = self._environment.observe()
            for i in range(len(obs)):
                self._state[i][0:-1] = self._state[i][1:]
                self._state[i][-1] = obs[i]

            V, action, reward = self._step()
            self._VsOnLastEpisode.append(V)
            isTerminal = self._environment.inTerminalState()
            if self._mode != -1:
                self._totalModeReward += reward
                
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
        reward = self._environment.act(action)

        return V, action, reward

    def _addSample(self, ponctualObs, action, reward, isTerminal):
        if self._mode != -1:
            self._tmpDataSet.addSample(ponctualObs, action, reward, isTerminal)
        else:
            self._dataSet.addSample(ponctualObs, action, reward, isTerminal)


    def _chooseAction(self):
        
        if self._mode != -1:
            action, V = self.bestAction()
        else:
            if self._dataSet.nElems() > self._replayMemoryStartSize:
                # e-Greedy policy
                if self._randomState.rand() < self._epsilon:
                    action = self._randomState.randint(0, self._environment.nActions())
                    V = 0
                else:
                    action, V = self.bestAction()
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
    """A replay memory consisting of circular buffers for observations, actions, rewards and terminals."""

    def __init__(self, env, randomState=None, maxSize=1000):
        """Initializer.

        Parameters:
            inputDims - For each subject i, inputDims[i] is a tuple where the first value is the memory size for this 
                subject and the rest describes the shape of each single observation on this subject (number, vector or 
                matrix). See base_classes.Environment.inputDimensions() documentation for more info about this format.
            randomState - Numpy random number generator. If None, a new one is created with default numpy seed.
            maxSize - The replay memory maximum size.
        """

        self._batchDimensions = env.inputDimensions()
        self._maxHistorySize = np.max([self._batchDimensions[i][0] for i in range (len(self._batchDimensions))])
        self._size = maxSize
        self._actions      = CircularBuffer(maxSize, dtype="int8")
        self._rewards      = CircularBuffer(maxSize)
        self._terminals    = CircularBuffer(maxSize, dtype="bool")
        self._observations = np.zeros(len(self._batchDimensions), dtype='object')
        # Initialize the observations container if necessary
        for i in range(len(self._batchDimensions)):
            self._observations[i] = CircularBuffer(maxSize, elemShape=self._batchDimensions[i][1:], dtype=env.observationType(i))

        if (randomState == None):
            self._randomState = np.random.RandomState()
        else:
            self._randomState = randomState

        self._nElems  = 0

    def actions(self):
        """Get all actions currently in the replay memory, ordered by time where they were taken."""

        return self._actions.getSlice(0)

    def rewards(self):
        """Get all rewards currently in the replay memory, ordered by time where they were received."""

        return self._rewards.getSlice(0)

    def terminals(self):
        """Get all terminals currently in the replay memory, ordered by time where they were observed.
        
        terminals[i] is True if actions()[i] lead to a terminal state (i.e. corresponded to a terminal 
        transition), and False otherwise.
        """

        return self._terminals.getSlice(0)

    def observations(self):
        """Get all observations currently in the replay memory, ordered by time where they were observed.
        
        observations[s][i] corresponds to the observation made on subject s before the agent took actions()[i].
        """

        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input].getSlice(0)

        return ret
            

    def randomBatch(self, size):
        """Return corresponding states, actions, rewards, terminal status, and next_states for size randomly 
        chosen transitions. Note that if terminal[i] == True, then next_states[s][i] == np.zeros_like(states[s][i]) for 
        each subject s.
        
        Arguments:
            size - Number of transitions to return.

        Returns:
            states - An ndarray(size=number_of_subjects, dtype='object), where states[s] is a 2+D matrix of dimensions
                size x s.memorySize x "shape of a given observation for this subject". States were taken randomly in 
                the data with the only constraint that they are complete regarding the histories for each observed 
                subject.
            actions - An ndarray(size=number_of_subjects, dtype='int32') where actions[i] is the action taken after 
                having observed states[:][i].
            rewards - An ndarray(size=number_of_subjects, dtype='float32') where rewards[i] is the reward obtained for 
                taking actions[i-1].
            next_states - Same structure than states, but next_states[s][i] is guaranteed to be the information 
                concerning the state following the one described by states[s][i] for each subject s.
            terminals - An ndarray(size=number_of_subjects, dtype='bool') where terminals[i] is True if actions[i] lead
                to terminal states and False otherwise
        Throws:
            SliceError - If a batch of this size could not be built based on current data set (not enough data or all 
                trajectories are too short).
        """

        rndValidIndices = np.zeros(size, dtype='int32')

        for i in range(size): # TODO: multithread this loop?
            rndValidIndices[i] = self._randomValidStateIndex()
            
        
        actions   = self._actions.getSliceBySeq(rndValidIndices)
        rewards   = self._rewards.getSliceBySeq(rndValidIndices)
        terminals = self._terminals.getSliceBySeq(rndValidIndices)
        states = np.zeros(len(self._batchDimensions), dtype='object')
        next_states = np.zeros_like(states)
        
        for input in range(len(self._batchDimensions)):
            states[input] = np.zeros((size,) + self._batchDimensions[input], dtype=self._observations[input].dtype)
            next_states[input] = np.zeros_like(states[input])
            for i in range(size):
                states[input][i] = self._observations[input].getSlice(rndValidIndices[i]+1-self._batchDimensions[input][0], rndValidIndices[i]+1)
                if rndValidIndices[i] >= self._nElems - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    next_states[input][i] = self._observations[input].getSlice(rndValidIndices[i]+2-self._batchDimensions[input][0], rndValidIndices[i]+2)

        return states, actions, rewards, next_states, terminals

    def _randomValidStateIndex(self):
        index_lowerBound = self._maxHistorySize - 1
        try:
            index = self._randomState.randint(index_lowerBound, self._nElems)
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
                if (i < 0 or self._terminals[i]):
                    break;

                i -= 1
                processed += 1
            
            if (processed < self._maxHistorySize - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < index_lowerBound):
                    startWrapped = True
                    index = self._nElems - 1
                if (startWrapped and index <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index
       

    def nElems(self):
        """Get the number of samples in this dataset (i.e. the current memory replay size)."""

        return self._nElems


    def addSample(self, obs, action, reward, isTerminal):
        """Store a (observation[for all subjects], action, reward, isTerminal) in the dataset. 

        Arguments:
            obs - An ndarray(dtype='object') where obs[s] corresponds to the observation made on subject s before the 
                agent took action [action].
            action - The action taken after having observed [obs].
            reward - The reward associated to taking this [action].
            isTerminal - Tells whether [action] lead to a terminal state (i.e. corresponded to a terminal transition).

        """        
        # Store observations
        for i in range(len(self._batchDimensions)):
            self._observations[i].append(obs[i])
        
        # Store rest of sample
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(isTerminal)

        if (self._nElems < self._size):
            self._nElems += 1

        
class CircularBuffer(object):
    def __init__(self, size, elemShape=(), extension=0.1, dtype="float32"):
        self._size = size
        self._data = np.zeros((int(size+extension*size),) + elemShape, dtype=dtype)
        self._trueSize = self._data.shape[0]
        self._lb   = 0
        self._ub   = size
        self._cur  = 0
        self.dtype = dtype
    
    def append(self, obj):
        if self._cur >= self._size:
            self._lb += 1
            self._ub += 1

        if self._ub >= self._trueSize:
            self._data[0:self._size-1] = self._data[self._lb+1:]
            self._lb  = 0
            self._ub  = self._size
            self._cur = self._size - 1
            
        self._data[self._cur] = obj

        self._cur += 1


    def __getitem__(self, i):
        return self._data[self._lb + i]

    def getSliceBySeq(self, seq):
        return self._data[seq + self._lb]

    def getSlice(self, start, end=sys.maxsize):
        if end == sys.maxsize:
            return self._data[self._lb+start:self._cur]
        else:
            return self._data[self._lb+start:self._lb+end]


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
