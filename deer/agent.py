""" 
This module contains classes used to define the standard behavior of the agent.
It relies on the controllers, the chosen training/test policy and the learning algorithm
to specify its behavior in the environment.

"""

import os
import numpy as np
import copy
import sys
import joblib
from warnings import warn

from .experiment import base_controllers as controllers
from .helper import tree 
from deer.policies import EpsilonGreedyPolicy

class NeuralAgent(object):
    """The NeuralAgent class wraps a learning algorithm (such as a deep Q-network) for training and testing in a given environment.
    
    Attach controllers to it in order to conduct an experiment (when to train the agent, when to test,...).
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent interacts
    learning_algo : object from class LearningAlgo
        The learning algorithm associated to the agent
    replay_memory_size : int
        Size of the replay memory. Default : 1000000
    replay_start_size : int
        Number of observations (=number of time steps taken) in the replay memory before starting learning. 
        Default: minimum possible according to environment.inputDimensions().
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    random_state : numpy random number generator
        Default : random seed.
    exp_priority : float
        The exponent that determines how much prioritization is used, default is 0 (uniform priority).
        One may check out Schaul et al. (2016) - Prioritized Experience Replay.
    train_policy : object from class Policy
        Policy followed when in training mode (mode -1)
    test_policy : object from class Policy
        Policy followed when in other modes than training (validation and test modes)
    only_full_history : boolean
        Whether we wish to train the neural network only on full histories or we wish to fill with zeroes the 
        observations before the beginning of the episode
    """

    def __init__(self, environment, learning_algo, replay_memory_size=1000000, replay_start_size=None, batch_size=32, random_state=np.random.RandomState(), exp_priority=0, train_policy=None, test_policy=None, only_full_history=True):
        inputDims = environment.inputDimensions()
        
        if replay_start_size == None:
            replay_start_size = max(inputDims[i][0] for i in range(len(inputDims)))
        elif replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))) :
            raise AgentError("Replay_start_size should be greater than the biggest history of a state.")
        
        self._controllers = []
        self._environment = environment
        self._learning_algo = learning_algo
        self._replay_memory_size = replay_memory_size
        self._replay_start_size = replay_start_size
        self._batch_size = batch_size
        self._random_state = random_state
        self._exp_priority = exp_priority
        self._only_full_history = only_full_history
        self._dataset = DataSet(environment, max_size=replay_memory_size, random_state=random_state, use_priority=self._exp_priority, only_full_history=self._only_full_history)
        self._tmp_dataset = None # Will be created by startTesting() when necessary
        self._mode = -1
        self._total_mode_reward = 0
        self._training_loss_averages = []
        self._Vs_on_last_episode = []
        self._in_episode = False
        self._selected_action = -1
        self._state = []
        for i in range(len(inputDims)):
            self._state.append(np.zeros(inputDims[i], dtype=float))
        if (train_policy==None):
            self._train_policy = EpsilonGreedyPolicy(learning_algo, environment.nActions(), random_state, 0.1)
        else:
            self._train_policy = train_policy
        if (test_policy==None):
            self._test_policy = EpsilonGreedyPolicy(learning_algo, environment.nActions(), random_state, 0.)
        else:
            self._test_policy = test_policy
        self.gathering_data=True    # Whether the agent is gathering data or not
        self.sticky_action=1        # Number of times the agent is forced to take the same action as part of one actual time step

    def setControllersActive(self, toDisable, active):
        """ Activate controller
        """
        for i in toDisable:
            self._controllers[i].setActive(active)

    def setLearningRate(self, lr):
        """ Set the learning rate for the gradient descent
        """
        self._learning_algo.setLearningRate(lr)

    def learningRate(self):
        """ Get the learning rate
        """
        return self._learning_algo.learningRate()

    def setDiscountFactor(self, df):
        """ Set the discount factor
        """
        self._learning_algo.setDiscountFactor(df)

    def discountFactor(self):
        """ Get the discount factor
        """
        return self._learning_algo.discountFactor()

    def overrideNextAction(self, action):
        """ Possibility to override the chosen action. This possibility should be used on the signal OnActionChosen.
        """
        self._selected_action = action

    def avgBellmanResidual(self):
        """ Returns the average training loss on the epoch
        """
        if (len(self._training_loss_averages) == 0):
            return -1
        return np.average(self._training_loss_averages)

    def avgEpisodeVValue(self):
        """ Returns the average V value on the episode (on time steps where a non-random action has been taken)
        """
        if (len(self._Vs_on_last_episode) == 0):
            return -1
        if(np.trim_zeros(self._Vs_on_last_episode)!=[]):
            return np.average(np.trim_zeros(self._Vs_on_last_episode))
        else:
            return 0

    def totalRewardOverLastTest(self):
        """ Returns the average sum of rewards per episode and the number of episode
        """
        return self._total_mode_reward/self._totalModeNbrEpisode, self._totalModeNbrEpisode

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
        if self._in_episode:
            raise AgentError("Trying to start mode while current episode is not yet finished. This method can be "
                             "called only *between* episodes for testing and validation.")
        elif mode == -1:
            raise AgentError("Mode -1 is reserved and means 'training mode'; use resumeTrainingMode() instead.")
        else:
            self._mode = mode
            self._total_mode_reward = 0.
            del self._tmp_dataset
            self._tmp_dataset = DataSet(self._environment, self._random_state, max_size=self._replay_memory_size, only_full_history=self._only_full_history)

    def resumeTrainingMode(self):
        self._mode = -1

    def summarizeTestPerformance(self):
        if self._mode == -1:
            raise AgentError("Cannot summarize test performance outside test environment.")

        self._environment.summarizePerformance(self._tmp_dataset, self._learning_algo, train_data_set=self._dataset)

    def train(self):
        """
        This function selects a random batch of data (with self._dataset.randomBatch) and performs a 
        Q-learning iteration (with self._learning_algo.train).        
        """
        # We make sure that the number of elements in the replay memory
        # is strictly superior to self._replay_start_size before taking 
        # a random batch and perform training
        if self._dataset.n_elems <= self._replay_start_size:
            return

        try:
            if hasattr(self._learning_algo, 'nstep'):
                observations, actions, rewards, terminals, rndValidIndices = self._dataset.randomBatch_nstep(self._batch_size, self._learning_algo.nstep, self._exp_priority)
                loss, loss_ind = self._learning_algo.train(observations, actions, rewards, terminals)
            else:
                states, actions, rewards, next_states, terminals, rndValidIndices = self._dataset.randomBatch(self._batch_size, self._exp_priority)
                loss, loss_ind = self._learning_algo.train(states, actions, rewards, next_states, terminals)

            self._training_loss_averages.append(loss)
            if (self._exp_priority):
                self._dataset.updatePriorities(pow(loss_ind,self._exp_priority)+0.0001, rndValidIndices[1])

        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

    def dumpNetwork(self, fname, nEpoch=-1):
        """ Dump the network
        
        Parameters
        -----------
        fname : string
            Name of the file where the network will be dumped
        nEpoch : int
            Epoch number (Optional)
        """
        try:
            os.mkdir("nnets")
        except Exception:
            pass
        basename = "nnets/" + fname

        for f in os.listdir("nnets/"):
            if fname in f:
                os.remove("nnets/" + f)

        all_params = self._learning_algo.getAllParams()

        if (nEpoch>=0):
            joblib.dump(all_params, basename + ".epoch={}".format(nEpoch))
        else:
            joblib.dump(all_params, basename, compress=True)

    def setNetwork(self, fname, nEpoch=-1):
        """ Set values into the network
        
        Parameters
        -----------
        fname : string
            Name of the file where the values are
        nEpoch : int
            Epoch number (Optional)
        """

        basename = "nnets/" + fname

        if (nEpoch>=0):
            all_params = joblib.load(basename + ".epoch={}".format(nEpoch))
        else:
            all_params = joblib.load(basename)

        self._learning_algo.setAllParams(all_params)

        
    def run(self, n_epochs, epoch_length):
        """
        This function encapsulates the inference and the learning.
        If the agent is in train mode (mode = -1):
            It starts by calling the controllers method "onStart", 
            Then it runs a given number of epochs where an epoch is made up of one or many episodes (called with 
            agent._runEpisode) and where an epoch ends up after the number of steps reaches the argument "epoch_length".
            It ends up by calling the controllers method "end".
        If the agent is on non train mode (mode > -1):
            This function runs a number of epochs in non train mode (mode > -1), thus without controllers.

        Parameters
        -----------
        n_epochs : int
            number of epochs
        epoch_length : int
            maximum number of steps for a given epoch
        """
        if(self._mode==-1):
            self._run_train(n_epochs, epoch_length)
        else:
            self._run_non_train(n_epochs, epoch_length)
            
    def _run_train(self, n_epochs, epoch_length):
        """
        This function encapsulates the whole process of the learning.
        It starts by calling the controllers method "onStart", 
        Then it runs a given number of epochs where an epoch is made up of one or many episodes (called with 
        agent._runEpisode) and where an epoch ends up after the number of steps reaches the argument "epoch_length".
        It ends up by calling the controllers method "end".

        Parameters
        -----------
        n_epochs : int
            number of epochs
        epoch_length : int
            maximum number of steps for a given epoch
        """
        for c in self._controllers: c.onStart(self)
        i = 0
        while i < n_epochs:
            self._training_loss_averages = []
            while epoch_length > 0: # run new episodes until the number of steps left for the epoch has reached 0
                epoch_length = self._runEpisode(epoch_length)
            i += 1
            for c in self._controllers: c.onEpochEnd(self)
            
        self._environment.end()
        for c in self._controllers: c.onEnd(self)

    def _run_non_train(self, n_epochs, epoch_length):
        """
        This function runs a number of epochs in non train mode (id > -1).

        Parameters
        -----------
        n_epochs : int
            number of epochs
        epoch_length : int
            maximum number of steps for a given epoch
        """
        for c in self._controllers: c.onStart(self)
        i = 0
        while i < n_epochs:
            self._totalModeNbrEpisode=0
            while epoch_length > 0:
                self._totalModeNbrEpisode += 1
                epoch_length = self._runEpisode(epoch_length)
            i += 1
            for c in self._controllers: c.onEpochEnd(self)
        
        self._environment.end()
        for c in self._controllers: c.onEnd(self)

    def _runEpisode(self, maxSteps):
        """
        This function runs an episode of learning. An episode ends up when the environment method "inTerminalState" 
        returns True (or when the number of steps reaches the argument "maxSteps")
        
        Parameters
        -----------
        maxSteps : int
            maximum number of steps before automatically ending the episode
        """
        self._in_episode = True
        initState = self._environment.reset(self._mode)
        inputDims = self._environment.inputDimensions()
        for i in range(len(inputDims)):
            if inputDims[i][0] > 1:
                self._state[i][1:] = initState[i][1:]
        
        self._Vs_on_last_episode = []
        is_terminal=False
        reward=0
        while maxSteps > 0:
            maxSteps -= 1
            if(self.gathering_data==True or self._mode!=-1):
                obs = self._environment.observe()
                
                for i in range(len(obs)):
                    self._state[i][0:-1] = self._state[i][1:]
                    self._state[i][-1] = obs[i]
                
                V, action, reward = self._step()
                
                self._Vs_on_last_episode.append(V)
                if self._mode != -1:
                    self._total_mode_reward += reward
                
                is_terminal = self._environment.inTerminalState()   # If the transition ends up in a terminal state, mark transition as terminal
                                                                    # Note that the new obs will not be stored, as it is unnecessary.
                    
                if(maxSteps>0):
                    self._addSample(obs, action, reward, is_terminal)
                else:
                    self._addSample(obs, action, reward, True)      # If the episode ends because max number of steps is reached, mark the transition as terminal
            
            for c in self._controllers: c.onActionTaken(self)
            
            if is_terminal:
                break
            
        self._in_episode = False
        for c in self._controllers: c.onEpisodeEnd(self, is_terminal, reward)
        return maxSteps

        
    def _step(self):
        """
        This method is called at each time step and performs one action in the environment.

        Returns
        -------
        V : float
            Estimated value function of current state.
        action : int
            The id of the action selected by the agent.
        reward : float
            Reward obtained for the transition
        """

        action, V = self._chooseAction()
        reward=0
        for i in range(self.sticky_action):
            reward += self._environment.act(action)

        return V, action, reward

    def _addSample(self, ponctualObs, action, reward, is_terminal):
        if self._mode != -1:
            self._tmp_dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)
        else:
            self._dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)


    def _chooseAction(self):
        
        if self._mode != -1:
            # Act according to the test policy if not in training mode
            action, V = self._test_policy.action(self._state, mode=self._mode, dataset=self._dataset)
        else:
            if self._dataset.n_elems > self._replay_start_size:
                # follow the train policy
                action, V = self._train_policy.action(self._state, mode=None, dataset=self._dataset)     #is self._state the only way to store/pass the state?
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()
                
        for c in self._controllers: c.onActionChosen(self, action)
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

    def __init__(self, env, random_state=None, max_size=1000000, use_priority=False, only_full_history=True):
        """Initializer.
        Parameters
        -----------
        inputDims : list of tuples
            Each tuple relates to one of the observations where the first value is the history size considered for this
            observation and the rest describes the shape of each punctual observation (e.g., scalar, vector or matrix). 
            See base_classes.Environment.inputDimensions() documentation for more info.
        random_state : Numpy random number generator
            If None, a new one is created with default numpy seed.
        max_size : float
            The replay memory maximum size. Default : 1000000
        """

        self._batch_dimensions = env.inputDimensions()
        self._max_history_size = np.max([self._batch_dimensions[i][0] for i in range (len(self._batch_dimensions))])
        self._size = max_size
        self._use_priority = use_priority
        self._only_full_history = only_full_history
        if ( isinstance(env.nActions(),int) ):
            self._actions      = CircularBuffer(max_size, dtype="int8")
        else:
            self._actions      = CircularBuffer(max_size, dtype='object')
        self._rewards      = CircularBuffer(max_size)
        self._terminals    = CircularBuffer(max_size, dtype="bool")
        if (self._use_priority):
            self._prioritiy_tree = tree.SumTree(max_size) 
            self._translation_array = np.zeros(max_size)

        self._observations = np.zeros(len(self._batch_dimensions), dtype='object')
        # Initialize the observations container if necessary
        for i in range(len(self._batch_dimensions)):
            self._observations[i] = CircularBuffer(max_size, elemShape=self._batch_dimensions[i][1:], dtype=env.observationType(i))

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self.n_elems  = 0
        self.sticky_action=1        # Number of times the agent is forced to take the same action as part of one actual time step

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
        """

        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input].getSlice(0)

        return ret

    def updatePriorities(self, priorities, rndValidIndices):
        """
        """
        for i in range( len(rndValidIndices) ):
            self._prioritiy_tree.update(rndValidIndices[i], priorities[i])

    def randomBatch(self, batch_size, use_priority):
        """Returns a batch of states, actions, rewards, terminal status, and next_states for a number batch_size of randomly
        chosen transitions. Note that if terminal[i] == True, then next_states[s][i] == np.zeros_like(states[s][i]) for
        each s.
        
        Parameters
        -----------
        batch_size : int
            Number of transitions to return.
        use_priority : Boolean
            Whether to use prioritized replay or not

        Returns
        -------
        states : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
            States are taken randomly in the data with the only constraint that they are complete regarding the history size 
            for each observation.
        actions : numpy array of integers [batch_size]
            actions[i] is the action taken after having observed states[:][i].
        rewards : numpy array of floats [batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals : numpy array of booleans [batch_size] 
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Throws
        -------
            SliceError
                If a batch of this batch_size could not be built based on current data set (not enough data or all
                trajectories are too short).
        """

        if (self._max_history_size + self.sticky_action - 1 >= self.n_elems):
            raise SliceError(
                "Not enough elements in the dataset to create a "
                "complete state. {} elements in dataset; requires {}"
                .format(self.n_elems, self._max_history_size))

        if (self._use_priority):
            #FIXME : take into account the case where self._only_full_history is false
            rndValidIndices, rndValidIndices_tree = self._randomPrioritizedBatch(batch_size)
            if (rndValidIndices.size == 0):
                raise SliceError("Could not find a state with full histories")
        else:
            rndValidIndices = np.zeros(batch_size, dtype='int32')
            if (self._only_full_history):
                for i in range(batch_size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(self._max_history_size+self.sticky_action-1)
            else:
                for i in range(batch_size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(minimum_without_terminal=self.sticky_action)
                

        actions   = self._actions.getSliceBySeq(rndValidIndices)
        rewards   = self._rewards.getSliceBySeq(rndValidIndices)
        terminals = self._terminals.getSliceBySeq(rndValidIndices)
    
        states = np.zeros(len(self._batch_dimensions), dtype='object')
        next_states = np.zeros_like(states)
        # We calculate the first terminal index backward in time and set it 
        # at maximum to the value self._max_history_size+self.sticky_action-1
        first_terminals=[]
        for rndValidIndex in rndValidIndices:
            first_terminal=1
            while first_terminal<self._max_history_size+self.sticky_action-1:
                if (self._terminals[rndValidIndex-first_terminal]==True or first_terminal>rndValidIndex):
                    break 
                first_terminal+=1
            first_terminals.append(first_terminal)
            
        for input in range(len(self._batch_dimensions)):
            states[input] = np.zeros((batch_size,) + self._batch_dimensions[input], dtype=self._observations[input].dtype)
            next_states[input] = np.zeros_like(states[input])
            for i in range(batch_size):
                slice=self._observations[input].getSlice(rndValidIndices[i]-self.sticky_action+2-min(self._batch_dimensions[input][0],first_terminals[i]+self.sticky_action-1), rndValidIndices[i]+1)
                if (len(slice)==len(states[input][i])):
                    states[input][i] = slice
                else:
                    for j in range(len(slice)):
                        states[input][i][-j-1]=slice[-j-1]
                 # If transition leads to terminal, we don't care about next state
                if rndValidIndices[i] >= self.n_elems - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    slice=self._observations[input].getSlice(rndValidIndices[i]+2-min(self._batch_dimensions[input][0],first_terminals[i]+1), rndValidIndices[i]+2)
                    if (len(slice)==len(states[input][i])):
                        next_states[input][i] = slice
                    else:
                        for j in range(len(slice)):
                            next_states[input][i][-j-1]=slice[-j-1]
                    #next_states[input][i] = self._observations[input].getSlice(rndValidIndices[i]+2-min(self._batch_dimensions[input][0],first_terminal), rndValidIndices[i]+2)
        
        if (self._use_priority):
            return states, actions, rewards, next_states, terminals, [rndValidIndices, rndValidIndices_tree]
        else:
            return states, actions, rewards, next_states, terminals, rndValidIndices

    def randomBatch_nstep(self, batch_size, nstep, use_priority):
        """Return corresponding states, actions, rewards, terminal status, and next_states for a number batch_size of randomly
        chosen transitions. Note that if terminal[i] == True, then next_states[s][i] == np.zeros_like(states[s][i]) for
        each s.
        
        Parameters
        -----------
        batch_size : int
            Number of transitions to return.
        nstep : int
            Number of transitions to be considered for each element
        use_priority : Boolean
            Whether to use prioritized replay or not

        Returns
        -------
        states : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * (history size+nstep-1) * size of punctual observation (which is 2D,1D or scalar)]).
            States are taken randomly in the data with the only constraint that they are complete regarding the history size 
            for each observation.
        actions : numpy array of integers [batch_size, nstep]
            actions[i] is the action taken after having observed states[:][i].
        rewards : numpy array of floats [batch_size, nstep]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * (history size+nstep-1) * size of punctual observation (which is 2D,1D or scalar)]).
        terminals : numpy array of booleans [batch_size, nstep] 
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Throws
        -------
            SliceError
                If a batch of this size could not be built based on current data set (not enough data or all
                trajectories are too short).
        """

        if (self._max_history_size + self.sticky_action - 1 >= self.n_elems):
            raise SliceError(
                "Not enough elements in the dataset to create a "
                "complete state. {} elements in dataset; requires {}"
                .format(self.n_elems, self._max_history_size))

        if (self._use_priority):
            #FIXME : take into account the case where self._only_full_history is false
            rndValidIndices, rndValidIndices_tree = self._randomPrioritizedBatch(batch_size)
            if (rndValidIndices.size == 0):
                raise SliceError("Could not find a state with full histories")
        else:
            rndValidIndices = np.zeros(batch_size, dtype='int32')
            if (self._only_full_history):
                for i in range(batch_size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(self._max_history_size+self.sticky_action*nstep-1)
            else:
                for i in range(batch_size): # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(minimum_without_terminal=self.sticky_action*nstep)
                

        actions=np.zeros((batch_size,(nstep)*self.sticky_action), dtype=int)
        rewards=np.zeros((batch_size,(nstep)*self.sticky_action))
        terminals=np.zeros((batch_size,(nstep)*self.sticky_action))
        for i in range(batch_size):
            actions[i] = self._actions.getSlice(rndValidIndices[i]-self.sticky_action*nstep+1,rndValidIndices[i]+self.sticky_action)
            rewards[i] = self._rewards.getSlice(rndValidIndices[i]-self.sticky_action*nstep+1,rndValidIndices[i]+self.sticky_action)
            terminals[i] = self._terminals.getSlice(rndValidIndices[i]-self.sticky_action*nstep+1,rndValidIndices[i]+self.sticky_action)
        
        observations = np.zeros(len(self._batch_dimensions), dtype='object')
        # We calculate the first terminal index backward in time and set it 
        # at maximum to the value self._max_history_size+self.sticky_action-1
        first_terminals=[]
        for rndValidIndex in rndValidIndices:
            first_terminal=1
            while first_terminal<self._max_history_size+self.sticky_action*nstep-1:
                if (self._terminals[rndValidIndex-first_terminal]==True or first_terminal>rndValidIndex):
                    break 
                first_terminal+=1
            first_terminals.append(first_terminal)
            
        batch_dimensions=copy.deepcopy(self._batch_dimensions)
        for input in range(len(self._batch_dimensions)):
            batch_dimensions[input]=tuple( x + y for x, y in zip(self._batch_dimensions[input],(self.sticky_action*(nstep+1)-1,0,0)) )
            observations[input] = np.zeros((batch_size,) + batch_dimensions[input], dtype=self._observations[input].dtype)
            for i in range(batch_size):
                slice=self._observations[input].getSlice(rndValidIndices[i]-self.sticky_action*nstep+2-min(self._batch_dimensions[input][0],first_terminals[i]-self.sticky_action*nstep+1), rndValidIndices[i]+self.sticky_action+1)
                if (len(slice)==len(observations[input][i])):
                    observations[input][i] = slice
                else:
                    for j in range(len(slice)):
                        observations[input][i][-j-1]=slice[-j-1]
                 # If transition leads to terminal, we don't care about next state
                if terminals[i][-1]:#rndValidIndices[i] >= self.n_elems - 1 or terminals[i]:
                    observations[input][rndValidIndices[i]:rndValidIndices[i]+self.sticky_action+1] = 0
        
        if (self._use_priority):
            return observations, actions, rewards, terminals, [rndValidIndices, rndValidIndices_tree]
        else:
            return observations, actions, rewards, terminals, rndValidIndices


    def _randomValidStateIndex(self, minimum_without_terminal):
        """ Returns the index corresponding to a timestep that is valid
        """
        index_lowerBound = minimum_without_terminal - 1
        # We try out an index in the acceptable range of the replay memory
        index = self._random_state.randint(index_lowerBound, self.n_elems-1) 

        # Check if slice is valid wrt terminals
        # The selected index may correspond to a terminal transition but not 
        # the previous minimum_without_terminal-1 transition
        firstTry = index
        startWrapped = False
        while True:
            i = index-1
            processed = 0
            for _ in range(minimum_without_terminal-1):
                if (i < 0 or self._terminals[i]):
                    break;

                i -= 1
                processed += 1
            if (processed < minimum_without_terminal - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < index_lowerBound):
                    startWrapped = True
                    index = self.n_elems - 1
                if (startWrapped and index <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index
    
    def _randomPrioritizedBatch(self, batch_size):
        indices_tree = self._prioritiy_tree.getBatch(batch_size, self._random_state, self)
        indices_replay_mem=np.zeros(indices_tree.size,dtype='int32')
        for i in range(len(indices_tree)):
            indices_replay_mem[i]= int(self._translation_array[indices_tree[i]] \
                         - self._actions.getLowerBound())
        
        return indices_replay_mem, indices_tree

    def addSample(self, obs, action, reward, is_terminal, priority):
        """Store the punctual observations, action, reward, is_terminal and priority in the dataset. 
        Parameters
        -----------
        obs : ndarray
            An ndarray(dtype='object') where obs[s] corresponds to the punctual observation s before the
            agent took action [action].
        action :  int
            The action taken after having observed [obs].
        reward : float
            The reward associated to taking this [action].
        is_terminal : bool
            Tells whether [action] lead to a terminal state (i.e. corresponded to a terminal transition).
        priority : float
            The priority to be associated with the sample
        """        
        # Store observations
        for i in range(len(self._batch_dimensions)):
            self._observations[i].append(obs[i])

        # Update tree and translation table
        if (self._use_priority):
            index = self._actions.getIndex()
            if (index >= self._size):
                ub = self._actions.getUpperBound()
                true_size = self._actions.getTrueSize()
                tree_ind = index%self._size
                if (ub == true_size):
                    size_extension = true_size - self._size
                    # New index
                    index = self._size - 1
                    tree_ind = -1
                    # Shift translation array
                    self._translation_array -= size_extension + 1
                tree_ind = np.where(self._translation_array==tree_ind)[0][0]
            else:
                tree_ind = index

            self._prioritiy_tree.update(tree_ind)
            self._translation_array[tree_ind] = index

        # Store rest of sample
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(is_terminal)

        if (self.n_elems < self._size):
            self.n_elems += 1

        
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
        if self._cur > self._size:  #> instead of >=
            self._lb += 1
            self._ub += 1

        if self._ub >= self._trueSize:
            # Rolling array without copying whole array (for memory constraints)
            # basic command: self._data[0:self._size-1] = self._data[self._lb:] OR NEW self._data[0:self._size] = self._data[self._lb-1:]
            n_splits=10
            for i in range(n_splits):
                self._data[i*(self._size)//n_splits:(i+1)*(self._size)//n_splits] = self._data[(self._lb-1)+i*(self._size)//n_splits:(self._lb-1)+(i+1)*(self._size)//n_splits]
            self._lb  = 0
            self._ub  = self._size
            self._cur = self._size #OLD self._size - 1

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

    def getLowerBound(self):
        return self._lb

    def getUpperBound(self):
        return self._ub

    def getIndex(self):
        return self._cur

    def getTrueSize(self):
        return self._trueSize


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
