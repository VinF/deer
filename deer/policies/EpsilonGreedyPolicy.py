from ..base_classes import Policy


class EpsilonGreedyPolicy(Policy):
    """The policy acts greedily with probability :math:`1-\epsilon` and acts randomly otherwise.
    It is now used as a default policy for the neural agent.

    Parameters
    -----------
    epsilon : float
        Proportion of random steps
    """
    def __init__(self, q_network, n_actions, random_state, epsilon):
        Policy.__init__(self, q_network, n_actions, random_state)
        self._epsilon = epsilon

    def action(self, state):
        if self.random_state.rand() < self._epsilon:
            action, V = self.randomAction()
        else:
            action, V = self.bestAction(state)

        return action, V

    def setEpsilon(self, e):
        """ Set the epsilon used for :math:`\epsilon`-greedy exploration
        """
        self._epsilon = e

    def epsilon(self):
        """ Get the epsilon for :math:`\epsilon`-greedy exploration
        """
        return self._epsilon
