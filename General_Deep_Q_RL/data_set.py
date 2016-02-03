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
    def __init__(self, history_sizes, random_state, max_steps=1000):
        """Construct a DataSet.

        Arguments:
            num_elements_in_batch - 
            rng - initialized numpy random number generator, used to
            max_steps - 

        """
        self._history_sizes = history_sizes
        self._size = size
        self._states    = np.zeros(size, dtype='object')
        self._actions   = np.zeros(size, dtype='int32')
        self._rewards   = np.zeros(size, dtype=theano.config.floatX)
        self._terminals = np.zeros(size, dtype='bool')

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self._n_elems  = 0
        self._current = 0
        self._lastInsertedStateIndex = 0


    def nStates(self):
        """Return the number of *states* in this data set. Might be different than what nElems returns.

        """


    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and next_states for batch_size randomly chosen state transitions.
        Note that if terminal[-1] == True, then next_states[-1] == None.
        
        Arguments:
            batch_size - Number of elements in the batch

        Returns:
            states - A list of size batch_size, where each element is an ndarray(dtype='object') of size 'number of inputs'. For each input i, 
                     this element observation[i] is a 3D matrix where first dimension denotes the history index (if no history, = 1), and the 
                     two others can be used to represent actual data.
            actions - The actions taken in each of those states.
            rewards - The rewards obtained for taking these actions in those states.
            next_states - Same than states, but shifted to the left by 1 time unit. If terminal[-1] == True, then next_states[-1] == None.
            terminal - Whether these actions lead to terminal states.

        """
        start, end = _randomSlice(batch_size)
        if (end < start):
            end += self._n_elems

        actions   = self._actions.take(range(start, end), mode='wrap')
        rewards   = self._rewards.take(range(start, end), mode='wrap')
        terminals = self._terminals.take(range(start, end), mode='wrap')

        states = self._states.take(range(start, end), mode='wrap')
        if (terminals[-1]):
            next_states = np.zeros_like(states)
            next_states[0:-1] = states[1:]
            next_states[-1] = None
        else:
            next_states = self._states.take(range(start+1, end+1), mode='wrap')

        return states, actions, rewards, next_states, terminals
       

    def nElems(self):
        """Return the number of *complete* samples in this data set (i.e. complete tuples (state, action, reward, isTerminal)).
        Might thus be different than what nStates returns.

        """
        return self._n_elems

    def lastRecordedState(self):
        """Return the last state inserted in this data set, either using addState or addSample.

        """
        return self._states[self._lastInsertedStateIndex];


    def addState(self, state):
        """Store an observation in the dataset. If the buffer is full, it uses a least recently used (LRU) approach 
        to replace an old element by the new one. Note that two subsequent calls to this function is prohibited; they
        should be interleaved with a call to addActionRewardTerminal.

        Arguments:
            state - An ndarray(dtype='object') whose length is the number of inputs.
                    For each input (so for each i in 0 <= i < M), observation[i] is a 3D matrix where 
                    first dimension denotes the history index (if no history, = 1), and the two others 
                    can be used to represent actual data.

        """
        
        self._states[self._current] = state
        self._lastInsertedStateIndex = self._current


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
        if (self._n_elems < self._size):
            self._n_elems += 1


    def addSample(self, state, action, reward, isTerminal):
        """Store a (state, action, reward, isTerminal) in the dataset. 
        Equivalent to 'addState(state); addActionRewardTerminal(action, reward, isTerminal)'.

        Arguments:
            state - An ndarray(dtype='object') whose length is the number of inputs.
                    For each input (so for each i in 0 <= i < M), observation[i] is a 3D matrix where 
                    first dimension denotes the history index (if no history, = 1), and the two others 
                    can be used to represent actual data.
            action - The id of the action taken in the last inserted state using addState.
            reward - The reward associated to taking 'action' in the last inserted state using addState.
            isTerminal - Tells whether 'action' lead to a terminal state (i.e. whether the tuple (state, action, reward, isTerminal) marks the end of a trajectory).

        """
        addState(state)
        addActionRewardTerminal(action, reward, isTerminal)
        

    def _randomSlice(self, size):
        """Get start and end indices of a random contiguous batch from the dataset. Two objects are considered "contiguous" if
        1) they are next to each other in the dataset, i.e. at position i and (i+1) % size
        2) self._terminals[i] == False.

        Arguments:
            size - The size of the slice.

        """
        if (size > self._n_elems):
            raise SliceError("Not enough elements in the buffer to sustain a slice of size " + size)

        slice = np.zeros(size, dtype=_buffer.dtype)

        start = self._random_state.randint(0, self._n_elems)
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
                if (i >= self._n_elems):
                    i = 0
            
            if (processed < size - 1):
                # if we stopped prematurely, shift slice to the left and try again
                end   = i + 1
                start = end - size
                if (start < 0):
                    startWrapped = True
                    start = self._n_elems + start
                if (startWrapped and start <= firstTry):
                    raise SliceError("Could not find a slice of size " + size)
            else:
                # else slice was ok: return it
                return start, end # self._buffer.take(range(start, end), mode='wrap')
        
class CircularBuffer(object):
    """A circular buffer of objects.

    """
    def __init__(self, size, type=None, random_state=None):
        """Construct a circular buffer.

        Arguments:
            size - The maximum number of elements in the buffer before new elements erase the oldest one.
			type - The type of the numpy array underlying this circular buffer. 'None' is for using a python list.
            random_state - A numpy.random.RandomState object or None for the default numpy RandomState.

        """
        self._size = size

        if (type == None):
            self._buffer = np.zeros(size, dtype='object')
        else:
            self._buffer = np.zeros(size, dtype=type)

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self._n_elems  = 0
        self._current = 0
        
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
