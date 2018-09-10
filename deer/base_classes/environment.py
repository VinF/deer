"""
This module defines the base class for the environments.

"""

import numpy as np

class Environment(object): 
    """All your Environment classes should inherit this interface.
    
    The environment defines the dynamics and the reward signal that the agent observes when interacting with it.
    
    An agent sees at any time-step from the environment a collection of observable elements. Observing the environment 
    at time t thus corresponds to obtaining a punctual observation for each of these elements. According to the control 
    problem to solve, it might be useful for the agent to not only take action based on the current punctual observations 
    but rather on a collection of the last punctual observations. In this framework, it's the environment that defines 
    the number of each punctual observation to be considered.

    Different "modes" are used in this framework to allow the environment to have different dynamics and/or reward signal. 
    For instance, in training mode, only a part of the dynamics may be available so that it is possible to see how well 
    the agent generalizes to a slightly different one.
    """
               
    def reset(self, mode):
        """Resets the environment and put it in mode [mode]. This function is called when beginning every new episode. 
        
        The [mode] can be used to discriminate for instance between an agent which is training or trying to get a 
        validation or generalization score. The mode the environment is in should always be redefined by resetting the
        environment using this method, meaning that the mode should be preserved until the next call to reset().

        Parameters
        -----------
        mode : int
            The mode to put the environment into. Mode "-1" is reserved and always means "training".

        Returns
        -------
        Initialization of the pseudo state at the beginning of a new episode: list (of lists) with size given by inputDimensions
        """

        raise NotImplementedError()
        
    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier 
            included between 0 included and nActions() excluded.
        """

        raise NotImplementedError()

    def inputDimensions(self):
        """Gets the shape of the input space for this environment.
        
        This returns a list whose length is the number of observations in the environment. Each element of the list is a tuple: 
        the first integer is always the history size considered for this observation and the rest describes the shape of the 
        observation at a given time step. For instance:
        - () or (1,) means each observation at a given time step is a single scalar and the history size is 1 (= only current 
        observation)
        - (N,) means each observation at a given time step is a single scalar and the history size is N
        - (N, M) means each observation at a given time step is a vector of length M and the history size is N
        - (N, M1, M2) means each observation at a given time step is a 2D matrix with M1 rows and M2 columns and the history 
        size is N
        """

        raise NotImplementedError()

    def nActions(self):
        """Gets the number of different actions that can be taken on this environment.
        It can be either an integer in the case of a finite discrete number of actions 
        or it can be a list of couples [min_action_value,max_action_value] for a continuous action space"""

        raise NotImplementedError()

    def inTerminalState(self):
        """Tells whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).

        As the majority of control tasks considered have no end (a continuous control should be operated), by default 
        this returns always False. But in the context of a video game for instance, terminal states can happen and in
        these cases, this method should be overridden.
        
        Returns
        -------
        isTerminal : bool
            Whether or not the current state is terminal
        """

        return False

    def observe(self):
        """Gets a list of punctual observations composing this environment.
        
        This returns a list where element i is a punctual observation. Note that the history  of observations is not 
        returned and only the current observation is.

        See the documentation of inputDimensions() for more information about the shape of the observations.
        """

        raise NotImplementedError()

    def summarizePerformance(self, test_data_set, *args, **kwargs):
        """Optional hook that can be used to show a summary of the performance of the agent on the
        environment in the current mode.

        Parameters
        -----------
        test_data_set : agent.DataSet 
            The dataset maintained by the agent in the current mode, which contains 
            observations, actions taken and rewards obtained, as well as wether each transition was terminal or 
            not. Refer to the documentation of agent.DataSet for more information.
        """

        pass

    def observationType(self, subject):
        """Gets the most inner type (np.uint8, np.float32, ...) of [subject].

        Parameters
        -----------
        subject : int
            The subject
        """

        return np.float32

    def end(self):
        """Optional hook called at the end of all epochs
        """

        pass
