"""
.. Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
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
        """Reset the environment and put it in mode [mode].
        
        The [mode] can be used to discriminate for instance between an agent which is training or trying to get a 
        validation or generalization score. The mode the environment is in should always be redefined by resetting the
        environment using this method, meaning that the mode should be preserved until the next call to reset().

        Parameters
        -----------
        mode : int
            The mode to put the environment into. Mode "-1" is reserved and always means "training".
        """

        raise NotImplementedError()
        
    def act(self, action):
        """Apply the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier 
            included between 0 included and nActions() excluded.
        """

        raise NotImplementedError()

    def inputDimensions(self):
        """Get the shape of the input space for this environment.
        
        This returns a list whose length is the number of subjects observed on the environment. Each element of the 
        list is a tuple whose content and size depends on the type of data observed: the first integer is always the 
        history size (or batch size) for observing this subject and the rest describes the shape of a single 
        observation on this subject:
        - () or (1,) means each observation on this subject is a single number and the history size is 1 (= no history)
        - (N,) means each observation on this subject is a single number and the history size is N
        - (N, M) means each observation on this subject is a vector of length M  and the history size is N
        - (N, M1, M2) means each observation on this subject is a matrix with M1 rows and M2 columns and the history 
        size is N
        """

        raise NotImplementedError()

    def nActions(self):
        """Get the number of different actions that can be taken on this environment."""

        raise NotImplementedError()

    def inTerminalState(self):
        """Tell whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).

        As the majority of control tasks considered have no end (a continuous control should be operated), by default 
        this returns always False. But in the context of a video game for instance, terminal states can occurs and 
        these cases this method should be overriden.
        
        Returns
        -------
        isTerminal : bool

        """

        return False

    def observe(self):
        """Get a list of punctual observations on all subjects composing this environment.
        
        This returns a list where element i is a punctual observation on subject i. You will notice that the history 
        of observations on this subject is not returned; only the very last observation. Each element is thus either 
        a number, vector or matrix and not a succession of numbers, vectors and matrices.

        See the documentation of batchDimensions() for more information about the shape of the observations according 
        to their mathematical representation (number, vector or matrix).
        """

        raise NotImplementedError()

    def summarizePerformance(self, test_data_set):
        """Additional optional hook that can be used to show a summary of the performance of the agent on the 
        environment in the current mode (in validation and or generalization for example).

        Parameters
        -----------
        test_data_set : agent.DataSet 
            The dataset maintained by the agent in the current mode, which contains 
            observations, actions taken and rewards obtained, as well as wether each transition was terminal or 
            not. Refer to the documentation of agent.DataSet for more information.
        """

        pass

    def observationType(self, subject):
        """Get the most inner type (np.uint8, np.float32, ...) of [subject].

        Parameters
        -----------
        subject : int
            The subject
        """

        return np.float32
