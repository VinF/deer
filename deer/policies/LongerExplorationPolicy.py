""" Exploration policy for permutation invariant environments
Authors: Vincent Francois-Lavet, Adrien Couetoux
"""

from ..base_classes import Policy
import itertools
import random
import copy
import numpy as np

class LongerExplorationPolicy(Policy):
    """Simple alternative to :math:`\epsilon`-greedy that can explore more
    efficiently for a broad class of realistic problems.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    length : int
        Length of the exploration sequences that will be considered
    """
    def __init__(self, q_network, n_actions, random_state, epsilon, length=10):
        Policy.__init__(self, q_network, n_actions, random_state)
        self._epsilon = epsilon
        self._l = length
        self._count_down = -1
        self._action_sequence = []

    def action(self, state):
        if self._count_down >= 0:
            # Take the next exploration action in the sequence
            V = 0
            action = self._action_sequence[self._count_down]
            self._count_down -= 1
        else:
            if self.random_state.rand() < self._epsilon/((1+(self._l-1)*(1-self._epsilon))):
                # Take a random action and build an exploration sequence for the next steps
                self._count_down = self._l - 1
                self._action_sequence = self.sampleUniformActionSequence()
                action = self._action_sequence[self._count_down]
                V = 0
                self._count_down -= 1
            else:
                # Simply act greedily with respect to what is currently believed to be the best action
                action, V = self.bestAction(state)
        
        return np.array(action), V

    def setEpsilon(self, e):
        """ Set the epsilon
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon
        """
        return self._epsilon

    def sampleUniformActionSequence(self):
        if ( isinstance(self.n_actions,int)):
            """ Sample an action sequence of length self._l, where the unordered sequences have uniform probabilities"""
            actions_list = range(self.n_actions)
        else:   
            """For N exploration steps, the goal is to have actions such that their sum spans quite uniformly 
            the whole range of possibilities. Among those possibilities, random choice/order of actions. """
            
            possible_actions=[]
            # Add for all actions N random element between min and max
            N=3
            for i,a in enumerate(self.n_actions):
                possible_actions.append([])
                for j in range(N):
                    possible_actions[i].append( self.random_state.uniform(self.n_actions[i][0],self.n_actions[i][1]) )
            actions_list = list(itertools.product(*possible_actions))
            
        sequences_with_replacement = list(itertools.combinations_with_replacement(actions_list, self._l))
        index_pick = self.random_state.randint(0, len(sequences_with_replacement))
        sequence = list(sequences_with_replacement[index_pick])
        self.random_state.shuffle(sequence)
        
        return sequence
