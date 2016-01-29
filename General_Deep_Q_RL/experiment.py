"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in any given environment.

Modifications: Vincent Francois-Lavet

Inspired from Nathan Sprague (https://github.com/spragunr/deep_q_rl)
"""

import logging
import numpy as np
import cv2
import data_set
import time


class MGExperiment(object):
    def __init__(self, agent, environment, 
                 num_epochs, epoch_length, test_length,
                 period_btw_summary_perfs, frame_skip, rng):
        self.agent = agent
        self.environment = environment
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.period_btw_summary_perfs=period_btw_summary_perfs
        self.frame_skip = frame_skip
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch and then the function
        get_summary_perf of the environment is called for all epochs 
        multiple of self.period_btw_summary_perfs
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)

            if self.test_length > 0:
                print "starting testing: epoch, self.test_length"
                self.agent.start_testing()
                print epoch, self.test_length
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)

                if (epoch%self.period_btw_summary_perfs==0):
                    self.environment.get_summary_perf(self.agent.data_set_test)

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps to execute.  An epoch is made of one or several
        episodes that end when a terminal state is encountered.

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training
        """
        if (self.agent.network.discount<0.99):
            self.agent.network.discount=1-(1-self.agent.network.discount)*self.agent.network.discount_inc
        self.agent.network.lr=self.agent.network.lr*self.agent.network.lr_dec
        
        if not testing:
            print "Epoch: "+str(epoch)
            print "Discount factor: "+str(self.agent.network.discount)
            print "Learning rate: "+str(self.agent.network.lr)
            print "Traning epsilon: "+str(self.agent.epsilon)

        steps_left = num_steps
        while steps_left > 0:
            _, num_steps = self.run_episode(steps_left, testing)
            steps_left -= num_steps

    def _init_episode(self, testing):
        """ Reset the environment and initialize the agent to start a new episode.
        
        Arguments:
        testing - Whether we are in test mode or not (boolean)
        
        Return: 
        observation - List of observed elements      
        """

        observation=self.environment.init(testing)
                
        self.agent.start_episode(observation)

        return observation

    def _step(self, action, testing):
        """ Repeat one action the appropriate number of times
        
        Arguments:
        action - Action to be taken
        testing - Whether we are in test mode or not (boolean)

        Return: 
        reward - Summed reward
        observation - New observation (list of observed elements)     
        terminal - Whether we are in a terminal state 
        """
        
        reward = 0
        for _ in range(self.frame_skip):
            reward_, observation, terminal = self.environment.act(action, testing)
            reward += reward_

        return reward, observation, terminal

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        Arguments:
        max_steps - Maximum number os steps for the episode
        testing - Whether we are in test mode or not (boolean)

        Return: 
        terminal - Boolean that indicates whether the episode ended 
        because the game ended or the agent died (True) or because the 
        maximum number of steps was reached (False).
        num_steps - num steps of the episode
        """

        observation = self._init_episode(testing)
        V_set=[]
        num_steps = 0
        while True:
                                    
            self.agent.add_sample_1(observation) ## NB : even if testing, it will just be a different queue
                        
            action, V = self.agent.step(observation)            
                        
            reward, observation, terminal = self._step(action, testing)

            self.agent.add_sample_2(reward) ## NB : even if testing, it will just be a different queue

            if not testing:
                self.agent.do_training()
            
            V_set.append(V)
            num_steps += 1
            
            if terminal or num_steps >= max_steps:
                if not testing:
                    print "Training loss (average bellman residual):"+str(np.average(self.agent.loss_averages))
                    print "Training average V value:"+str(np.average(V_set))
                
                break

        return terminal, num_steps
