"""
This module defines the base class for the policies.

"""

import numpy as np

class Policy(object):
    """Abstract class for all policies.
    A policy takes observations as input, and outputs an action.

    Parameters
    -----------
    learning_algo : object from class LearningALgo
    n_actions : int or list
        Definition of the action space provided by Environment.nActions()
    random_state : numpy random number generator
    """

    def __init__(self, learning_algo, n_actions,random_state):
        self.learning_algo = learning_algo
        self.n_actions = n_actions
        self.random_state = random_state

        pass

    def bestAction(self, state, mode=None, *args, **kwargs):
        """ Returns the best Action for the given state. This is an additional encapsulation for q-network.
        """
        action,V = self.learning_algo.chooseBestAction(state, mode, *args, **kwargs)
        return action, V

    def randomAction(self):
        """ Returns a random action
        """
        if ( isinstance(self.n_actions,int)):
            # Discrete set of actions [0,nactions[
            action = self.random_state.randint(0, self.n_actions)
        else:
            # Continuous set of actions
            action=[]
            for a in self.n_actions:
                action.append( self.random_state.uniform(a[0],a[1]) )
            action=np.array(action)

        V = 0
        return action, V


    def action(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()
