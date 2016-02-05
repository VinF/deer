"""This class stores all of the samples for training. 

Author: Vincent Francois-Lavet
Contributor: David Taralla
"""

import numpy as np
import theano
import copy 


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
        self._historySizes = history_sizes
        self._maxHistorySize = np.max(history_sizes)
        self._size = size
        self._observations = np.zeros(len(history_sizes), dtype='object') # One list per input; will be initialized at 
        self._actions      = np.zeros(size, dtype='int32')                # first call of addState
        self._rewards      = np.zeros(size, dtype=theano.config.floatX)
        self._terminals    = np.zeros(size, dtype='bool')

        if (randomState == None):
            self._randomState = np.random.RandomState()
        else:
            self._randomState = randomState

        self._nElems  = 0


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

        rndValidIndices = np.zeros(batch_size)
        for i in range(batch_size): # TODO: multithread this loop?
            rndValidIndices[i] = _randomValidStateIndex()
            
        
        actions   = self._actions[rndValidIndices]
        rewards   = self._rewards[rndValidIndices]
        terminals = self._terminals[rndValidIndices]
        states = np.zeros(len(history_sizes), dtype='object')
        next_states = np.zeros_like(states)

        for input in range(len(_historySizes)):
            states[input] = np.zeros((batch_size, self._historySizes[input],) + np.array(state[i]).shape)
            for i in range(batch_size):
                states[input][i] = self._observations[input][rndValidIndices[i]-self._historySizes[input]:rndValidIndices[i]]
                if rndValidIndices[i] >= self._nElems - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    next_states[input][i] = self._observations[input][rndValidIndices[i]+1-self._historySizes[input]:rndValidIndices[i]+1]

        
        return states, actions, rewards, next_states, terminals

    def _randomValidStateIndex(self):
        index = self._randomState.randint(self._maxHistorySize-1, self._nElems)
        
        # Check if slice is valid wrt terminals
        firsTry = index
        startWrapped = False
        while True:
            i = index-1
            for processed in range(1, self._maxHistorySize):
                if (i < 0 or self._terminals[i]):
                    break;

                i -= 1
            
            if (processed < self._maxHistorySize - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < 0):
                    startWrapped = True
                    index = self._nElems - 1
                if (startWrapped and start <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index
       

    def nElems(self):
        """Return the number of *complete* samples in this data set (i.e. complete tuples (state, action, reward, isTerminal)).
        Might thus be different than what nStates returns.

        """
        return self._nElems


    def addPonctualObservation(self, ponctualObs):
        """Store an observation in the dataset. If the buffer is full, it uses a least recently used (LRU) approach 
        to replace an old element by the new one. Note that two subsequent calls to this function is prohibited; they
        should be interleaved with a call to addActionRewardTerminal.

        Arguments:
            ponctualObs - An ndarray(dtype='object') whose length is the number of inputs.
                          For each input i, ponctualObs[i] is a 2D matrix that represents the actual data.

        """

        # Initialize the observations container if necessary
        if (self._nElems == 0):
            for i in range(len(self._historySizes)):
                self._observations[i] = np.zeros((self._size,) + np.array(ponctualObs[i]).shape)
        
        # Store observations
        for i in range(len(self._historySizes)):
            self._observations[i] = np.roll(self._observations[i], -1)
            self._observations[i][-1] = ponctualObs[i]


    def addActionRewardTerminal(self, action, reward, isTerminal):
        """Store the (action, reward, isTerminal) tuple relative to the last inserted state (through addState) in the 
        dataset.
        Note that two subsequent calls to this function is prohibited; they should be interleaved with a call to addState.

        Arguments:
            action - The id of the action taken in the last inserted state using addState.
            reward - The reward associated to taking 'action' in the last inserted state using addState.
            isTerminal - Tells whether 'action' lead to a terminal state (i.e. whether the tuple (state, action, reward, 
                         isTerminal) marks the end of a trajectory).

        """
        self._actions = np.roll(self._actions, -1)
        self._rewards = np.roll(self._rewards, -1)
        self._terminals = np.roll(self._terminals, -1)
        self._actions[-1] = action;
        self._rewards[-1] = reward;
        self._terminals[-1] = isTerminal;

        if (self._nElems < self._size):
            self._nElems += 1


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
        addState(ponctualObs)
        addActionRewardTerminal(action, reward, isTerminal)
        

    def _randomSlice(self, size):
        """Get start and end indices of a random contiguous batch from the dataset. Two objects at i and j are 
        considered "contiguous" if
        1) they are next to each other in the dataset, i.e. at position i and j=i+1
        2) self._terminals[i] == False.

        Arguments:
            size - The size of the slice.

        """
        if (size > self._nElems):
            raise SliceError("Not enough elements in the buffer to sustain a slice of size " + size)

        slice = np.zeros(size, dtype=_buffer.dtype)

        start = self._randomState.randint(0, self._nElems - size)
        end   = start + size
        
        # Check if slice is valid wrt terminals
        firsTry = start
        startWrapped = False
        while True:
            i = start
            for processed in range(size):
                if (self._terminals[i]):
                    break;

                i += 1
                if (i >= self._nElems):
                    i = 0
            
            if (processed < size - 1):
                # if we stopped prematurely, shift slice to the left and try again
                end   = i + 1
                start = end - size
                if (start < 0):
                    startWrapped = True
                    end = self._nElems
                    start = end - size
                if (startWrapped and start <= firstTry):
                    raise SliceError("Could not find a slice of size " + size)
            else:
                # else slice was ok according to mask
                return start, end

        
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
