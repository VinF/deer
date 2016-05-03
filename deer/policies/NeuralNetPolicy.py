from ..base_classes import Policy


class NeuralNetPolicy(Policy):
    """The policy that acts greedily w.r.t. a given Q-network with probability 1-\epsilon, and acts randomly otherwise.
    It is now used as a default policy for the neural agent.
    """
    def __init__(self, environment_, q_network_, epsilon_, replay_memory_start_size_, random_state_):
        Policy.__init__(self, environment_)
        self.q_network = q_network_
        self.epsilon = epsilon_
        self.replay_memory_start_size = replay_memory_start_size_
        self.random_state = random_state_

    def best_action(self, state):
        """ Returns the best Action
        """
        action = self.q_network.chooseBestAction(state)
        V = max(self.q_network.qValues(state))
        return action, V

    def act(self, state):
        if self.random_state.rand() < self.epsilon:
            action = self.random_state.randint(0, self.environment.nActions())
            V = 0
        else:
            action, V = self.best_action(state)

        return action, V
