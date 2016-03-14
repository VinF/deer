"""This file defines the base Environment and QNetwork interfaces you should inherit from when creating new 
environments and new algorithms.

Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import numpy as np

class Environment(object): 
    """All your Environment classes should inherit this interface.
    
    An agent aways lies in a given environment. The environment surrounds the agent, and the agent can take actions
    on this environment that will change its state. At any time the environment can be observed by the agent. Its in 
    your environment that you will define the shape of your algorithm's input space.
    
    An environment can be seen as a collection of observable elements, named its *subjects*. Observing the environment 
    at time t thus corresponds to observing each of the subject at time t: your Environment class is the place where 
    you describe these subjects, i.e. the shape of the observations of these subjects. According to the control problem
    to solve, it might be useful for the agent to not only take action based on the current state of a subject 's' but 
    rather on the last 'n_s' observations of 's', the current one included. Such a memory size (or "batch size") is 
    also defined in your Environment class.

    For instance, let an environment E be a collection of 3 subjects S1, S2, S3. S1 is the view of a given camera, S2 
    the view of a second camera and S3 the hour of the day. If both camera views have W * H gray-scaled pixels, and if
    we want to take decisions based on the last 4 frames of each camera view and on the current hours of the day, it 
    means that the shape of the agent's input space will be [(4, W, H), (4, W, H), ()].

    Furthermore, the environment can be controlled so that experiments and simulations can be carried on. For instance,
    an environment can be reset to an initial state when the agent wants to start a new episode. The agent can act on 
    it (take an action on it) and ask it whether it reached a terminal state or not. However, as the actions one can
    take on the environment are closely related to the environment itself, these actions are defined as unique integers
    by your Environment class again, that should also be able to give the number of possible actions.

    Finally, an environment might have different behaviors based on the mode it is in. For instance, in training mode,
    the environment could only modify its state according to a part of a given database, while in a validation mode it 
    would modify its state according to the other part of this database.
    """
               
    def reset(self, mode):
        """Reset the environment and put it in mode [mode].
        
        The [mode] can be used to discriminate for instance between an agent which is training or trying to get a 
        validation or generalization score. The mode the environment is in should always be redefined by resetting the
        environment using this method, meaning that the mode should be preserved until the next call to reset().

        Parameters:
            mode [int] - The mode to put the environment into. Mode "-1" is reserved and always means "training".
        """

        raise NotImplementedError()
        
    def act(self, action):
        """Apply the agent action [action] on the environment.

        Parameters:
            action [int] - The action selected by the agent to operate on the environment. Should be an identifier 
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

        Parameters:
            test_data_set [agent.DataSet] - The dataset maintained by the agent in the current mode, which contains 
                observations, actions taken and rewards obtained, as well as wether each transition was terminal or 
                not. Refer to the documentation of agent.DataSet for more information.
        """

        pass

    def observationType(self, subject):
        """Get the most inner type (np.uint8, np.float32, ...) of [subject].

        Parameters:
            subject [int] - The subject
        """

        return np.float32

class QNetwork(object):        
    def train(self, states, actions, rewards, nextStates, terminals):
        raise NotImplementedError()

    def chooseBestAction(self, state):
        raise NotImplementedError()

    def qValues(self, state):
        raise NotImplementedError()

    def setLearningRate(self, lr):
        raise NotImplementedError()

    def setDiscountFactor(self, df):
        raise NotImplementedError()

    def learningRate(self):
        raise NotImplementedError()

    def discountFactor(self):
        raise NotImplementedError()

if __name__ == "__main__":
    pass
