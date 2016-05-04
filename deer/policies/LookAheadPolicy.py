from ..base_classes import Policy
import copy
import numpy as np


class LookAheadPolicy(Policy):
    """A policy that does a one step look ahead, then evaluates the new states with the Q-network of the agent.
    It then acts epsilon-greedily with respect to the observed states.
    It does a deep copy of the environment for every trajectory it explores, so the current version can be very slow.
    """

    NB_STEPS_LOOKAHEAD = 1

    def __init__(self, environment_, q_network, epsilon_, replay_memory_start_size_, random_state_, dataset_=None):
        Policy.__init__(self, environment_, dataset_)
        self.target_q_network = q_network
        self.epsilon = epsilon_
        self.replay_memory_start_size = replay_memory_start_size_
        self.random_state = random_state_
        self.training_loss_averages = []

    def best_action(self, state):
        """ Returns the best Action
        """
        print "in best action"
        values = np.zeros(self.environment.nActions())
        for action in range(self.environment.nActions()):
            print "in loop"
            temp_env = copy.deepcopy(self.environment)
            temp_env.act(action)
            new_state = temp_env.observe()
            values[action] = self.target_q_network.qValues(new_state)[action]
        action = np.argmax(values)
        V = max(self.target_q_network.qValues(state))
        return action, V

    def act(self, state):
        if self.random_state.rand() < self.epsilon:
            action = self.random_state.randint(0, self.environment.nActions())
            V = 0
        else:
            action, V = self.best_action(state)
        return action, V