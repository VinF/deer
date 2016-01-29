"""
The NeuralAgent class wraps a deep Q-network for training and testing
in any given environment.

Modifications: Vincent Francois-Lavet

Insipired from Nathan Sprague (https://github.com/spragunr/deep_q_rl)
"""

import os
import cPickle
import time
import logging
import numpy as np
import data_set
import copy
import sys
sys.setrecursionlimit(10000)


class NeuralAgent(object):

    def __init__(self, q_network, epsilon_start, epsilon_min,
                 epsilon_decay, replay_memory_size,
                 replay_start_size, update_frequency, batch_size, rng):

        self.network = q_network
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_memory_size = replay_memory_size
        self.replay_start_size = replay_start_size
        self.update_frequency = update_frequency
        self.batch_size=batch_size
        self.rng = rng
        self.num_elements_in_batch = self.network.num_elements_in_batch
        self.num_actions = self.network.num_actions

        self.data_set = data_set.DataSet(self.num_elements_in_batch,
                                             rng=rng,
                                             max_steps=self.replay_memory_size)
        self.data_set_test = data_set.DataSet(self.num_elements_in_batch,
                                             rng=rng,
                                             max_steps=self.replay_memory_size) #max(self.num_elements_in_batch)+32 ) #FIXME--> a smaller one

        self.epsilon = self.epsilon_start
        if self.epsilon_decay != 0:
            self.epsilon_rate = ((self.epsilon_start - self.epsilon_min) /
                                 self.epsilon_decay)
        else:
            self.epsilon_rate = 0

        self.testing = False

        self.holdout_data = None


    def start_episode(self, observation): #FIXMETT
        """
        This method is called once at the beginning of each episode.
        No reward is provided, because reward is only available after
        an action has been taken.

        Arguments:
           observation - height x width numpy array

        Returns:
           An integer action
        """

        self.step_counter = 0

        # We report the mean loss for every epoch.
        if self.testing:
            self.test_loss_averages = []            
        if not self.testing:
            self.loss_averages = []

        self.start_time = time.time()

        self.last_action = 0
        self.last_state = copy.copy(observation)


    def step(self, state):
        """
        This method is called each time step.

        Arguments:
           reward      - Real valued reward.
           observation - A height x width numpy array

        Returns:
           action - An integer action.
           V - Estimated value function of current state
        """

        V=0
        self.step_counter += 1
        #TESTING---------------------------
        if self.testing:
            if len(self.data_set_test) > self.replay_start_size:
                action, V = self._choose_action(0., self.data_set_test.get_one_state(self.last_index_test))
            else: # Still gathering initial data
                action=0#self.rng.randint(0, self.num_actions)

        #NOT TESTING---------------------------
        else:

            if len(self.data_set) > self.replay_start_size:
                self.epsilon = max(self.epsilon_min,
                                   self.epsilon - self.epsilon_rate)
                
                action, V = self._choose_action(self.epsilon, self.data_set.get_one_state(self.last_index) )
            else: # Still gathering initial data
                action=self.rng.randint(0, self.num_actions)
            

        self.last_action = copy.copy(action)
        self.last_state = copy.copy(state)
        
        return action, V


    def add_sample_1(self, observation):
        if not self.testing:
            self.last_index=self.data_set.add_sample_1(observation)
        else:
            self.last_index_test=self.data_set_test.add_sample_1(observation)

    def add_sample_2(self, reward):
        if not self.testing:
            self.data_set.add_sample_2(self.last_index, self.last_action, reward, False)
        else:
            self.data_set_test.add_sample_2(self.last_index_test, self.last_action, reward, False)
            self.total_test_reward+=reward

    def do_training(self):        
        """
        Peforms training if self.step_counter is a multiple of self.update_frequency and if the size of the 
        dataset is at least bigger than the size of a batch
        """
        if self.step_counter % self.update_frequency == 0 and len(self.data_set)>self.batch_size:
            loss = self._do_training() 
            self.loss_averages.append(loss)

    def _choose_action(self, epsilon, state):
        """
        Add the most recent data to the data set and choose

        Arguments:
           epsilon - float, exploration of the epsilon greedy
           state - a list of k elements for ONE belief state (e.g. 2D obs --> element is num_element*w*h)

        Returns:
           An integer - action based on the current policy
        """

        if self.rng.rand() < epsilon: # Random action
            action = self.rng.randint(0, self.num_actions)
        else: # Select best action according to the agent
            action = self.network.choose_best_action(state)

        return action, max(self.network.q_vals(state))

    def _do_training(self):
        """
        Returns the average loss for the current batch.
        May be overridden if a subclass needs to train the network
        differently.
        
        
        """
        states, actions, rewards, next_states, terminals = \
                                self.data_set.random_batch(
                                    self.network.batch_size)

        return self.network.train(states, actions, rewards,
                                  next_states, terminals)


    def start_testing(self):
        self.testing = True
        self.total_test_reward = 0
        # Reinit data_set_test
        self.data_set_test = data_set.DataSet(self.num_elements_in_batch,
                                             rng=self.rng,
                                             max_steps=self.replay_memory_size) #max(self.num_elements_in_batch)+32 ) #FIXME--> a smaller one

    def finish_testing(self, epoch):
        self.testing = False

        print "Testing score (epoch %d) is %d" % (epoch, self.total_test_reward)


if __name__ == "__main__":
    pass
