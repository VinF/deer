from ..base_classes import Policy


class NeuralNetPolicy(Policy):
    """The policy that acts greedily w.r.t. a given Q-network with probability 1-\epsilon, and acts randomly otherwise.
    It is now used as a default policy for the neural agent.
    """
    def __init__(self, q_network, n_actions, epsilon, random_state):
        Policy.__init__(self, q_network, n_actions, random_state)
        self.epsilon = epsilon

    def act(self, state):
        if self.random_state.rand() < self.epsilon:
            action = self.random_state.randint(0, self.n_actions)
            V = 0
        else:
            action, V = self.bestAction(state)

        return action, V
