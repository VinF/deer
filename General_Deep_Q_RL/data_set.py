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
    def __init__(self, history_sizes, randomState, maxSize=1000):
        """Construct a DataSet.

        Arguments:
            history_sizes - For each input i, history_sizes[i] is the size of the history for this input.
            randomState - Numpy random number generator. If None, a new one is created with default numpy seed.
            maxSize - The maximum size of this buffer.

        """
        self._historySizes = history_sizes
        self._maxHistorySize = np.max(history_sizes)
        self._size = size
        self._observations = np.zeros(size, dtype='object')
        self._actions      = np.zeros(size, dtype='int32')
        self._rewards      = np.zeros(size, dtype=theano.config.floatX)
        self._terminals    = np.zeros(size, dtype='bool')

        if (randomState == None):
            self._randomState = np.random.RandomState()
        else:
            self._randomState = randomState

        self._nElems  = 0
        self._current = 0


    def randomBatch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and next_states for batch_size randomly chosen state transitions.
        Note that if terminal[-1] == True, then next_states[-1] == None.
        
        Arguments:
            batch_size - Number of elements in the batch

        Returns:
            states - A list of size batch_size, where each element is an ndarray(dtype='object') 'A' of size 'number of inputs'. For each input i, 
                     A[i] is a 3D matrix where first dimension denotes the history index (if no history, = 1), and the 
                     two others can be used to represent actual data.
            actions - The actions taken in each of those states.
            rewards - The rewards obtained for taking these actions in those states.
            next_states - Same than states, but shifted to the left by 1 time unit. If terminal[-1] == True, then next_states[-1] == None.
            terminal - Whether these actions lead to terminal states.

        """
        start, end = _randomSlice(batch_size + self._maxHistorySize)
        start += self._maxHistorySize
        if (end < start):
            end += self._nElems

        
        actions   = self._actions.take(range(start, end), mode='wrap')
        rewards   = self._rewards.take(range(start, end), mode='wrap')
        terminals = self._terminals.take(range(start, end), mode='wrap')

        states = [None] * batch_size
        for i in range(batch_size):
            states[i] = np.zeros(len(self._historySizes), dtype='object')
            for j in range(len(self._historySizes)):
                states[i][j] = self._observations.take(range(start+i-self._historySizes[j], end), mode='wrap')
                
        next_states = [None] * batch_size
        next_states[0:-1] = states[1:]
        if (terminals[-1] == False):
            next_states[-1] = np.zeros(len(self._historySizes), dtype='object')
            for j in range(len(self._historySizes)):
                next_states[-1][j] = self._observations.take(range(end+1-self._historySizes[j], end+1), mode='wrap')

        return states, actions, rewards, next_states, terminals
       

    def nElems(self):
        """Return the number of *complete* samples in this data set (i.e. complete tuples (state, action, reward, isTerminal)).
        Might thus be different than what nStates returns.

        """
        return self._nElems


    def addObservation(self, observation):
        """Store an observation in the dataset. If the buffer is full, it uses a least recently used (LRU) approach 
        to replace an old element by the new one. Note that two subsequent calls to this function is prohibited; they
        should be interleaved with a call to addActionRewardTerminal.

        Arguments:
            observation - An ndarray(dtype='object') whose length is the number of inputs.
                          For each input i, observation[i] is a 2D matrix that represents the actual data.

        """
        
        self._observations[self._current] = observation


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
        self._actions[self._current] = action;
        self._rewards[self._current] = reward;
        self._terminals[self._current] = isTerminal;

        self._current += 1
        if (self._current >= self._size):
            self._current = 0
        if (self._nElems < self._size):
            self._nElems += 1


    def addSample(self, observation, action, reward, isTerminal):
        """Store a (observation, action, reward, isTerminal) in the dataset. 
        Equivalent to 'addState(observation); addActionRewardTerminal(action, reward, isTerminal)'.

        Arguments:
            observation - An ndarray(dtype='object') whose length is the number of inputs.
                          For each input (so for each i in 0 <= i < M), observation[i] is a 2D matrix that represents 
                          actual data.
            action - The id of the action taken in the last inserted state using addState.
            reward - The reward associated to taking 'action' in the last inserted state using addState.
            isTerminal - Tells whether 'action' lead to a terminal state (i.e. whether the tuple (state, action, reward, isTerminal) marks the end of a trajectory).

        """
        addState(observation)
        addActionRewardTerminal(action, reward, isTerminal)
        

    def _randomSlice(self, size):
        """Get start and end indices of a random contiguous batch from the dataset. Two objects are considered "contiguous" if
        1) they are next to each other in the dataset, i.e. at position i and (i+1) % size
        2) self._terminals[i] == False.

        Arguments:
            size - The size of the slice.

        """
        if (size > self._nElems):
            raise SliceError("Not enough elements in the buffer to sustain a slice of size " + size)

        slice = np.zeros(size, dtype=_buffer.dtype)

        start = self._randomState.randint(0, self._nElems)
        end   = start + size
        
        # Check if slice is valid wrt mask
        firstTry = start
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
                    start = self._nElems + start
                if (startWrapped and start <= firstTry):
                    raise SliceError("Could not find a slice of size " + size)
            else:
                # else slice was ok according to mask
                return start, end # self._buffer.take(range(start, end), mode='wrap')

        
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
