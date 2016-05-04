from ..base_classes import Policy
from deer.q_networks.q_net_theano import MyQNetwork
import numpy as np


class ExploringNetPolicy(Policy):
    """A policy that builds its own Q-network.
    This Q-network can be trained in any way we want; in this case, we chose to train it on the same dataset than
    the one seen by the agent, but replacing the rewards of (s,a,s') by sum(abs(s-s'))
    """

    UPDATE_RULE = 'deepmind_rmsprop'
    BATCH_ACCUMULATOR = 'sum'
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001  # .01
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    BATCH_SIZE = 32
    NETWORK_TYPE = "General_DQN_0"
    FREEZE_INTERVAL = 1000

    def __init__(self, environment_, q_network, epsilon_, replay_memory_start_size_, random_state_, dataset_=None):
        Policy.__init__(self, environment_, dataset_)
        self.target_q_network = q_network
        self.behavior_q_network = MyQNetwork(
            environment_,
            self.RMS_DECAY,
            self.RMS_EPSILON,
            self.MOMENTUM,
            self.CLIP_DELTA,
            self.FREEZE_INTERVAL,
            self.BATCH_SIZE,
            self.NETWORK_TYPE,
            self.UPDATE_RULE,
            self.BATCH_ACCUMULATOR,
            np.random.RandomState(123456))
        self.epsilon = epsilon_
        self.replay_memory_start_size = replay_memory_start_size_
        self.random_state = random_state_
        self.training_loss_averages = []

    def best_action(self, state):
        """ Returns the best Action
        """
        action = self.behavior_q_network.chooseBestAction(state)
        V = max(self.behavior_q_network.qValues(state))
        return action, V

    def act(self, state):
        if self.random_state.rand() < self.epsilon:
            action = self.random_state.randint(0, self.environment.nActions())
            V = 0
        else:
            action, V = self.best_action(state)

        return action, V

    def train(self):
        if self.dataset.nElems() < self.replay_memory_start_size:
            return

        states, actions, rewards, next_states, terminals = self.dataset.randomBatch(self.BATCH_SIZE)
        exploration_rewards = np.zeros(rewards.shape)
        for subject in range(np.shape(states)[0]):
            for batch_item in range(np.shape(states[subject])[0]):
                exploration_rewards[batch_item] += np.sum( np.absolute(np.ravel(states[subject][batch_item]) - np.ravel(next_states[subject][batch_item])) )
        loss = self.behavior_q_network.train(states, actions, exploration_rewards, next_states, terminals)
        self.training_loss_averages.append(loss)
        # print str(states[0][0])
        # print str(states[1][0])
        # print str(next_states[0][0])
        # print str(next_states[1][0])
        # print str(exploration_rewards[0])
        # print "loss after epoch " + str(loss)

    def update_after_action(self):
        self.train()

    def update_after_epoch(self):
        self.train()