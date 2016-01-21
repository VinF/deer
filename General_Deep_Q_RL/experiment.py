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
                 period_btw_summary_perfs, frame_skip, max_start_nullops, rng):
        self.agent = agent
        self.environment = environment
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.period_btw_summary_perfs=period_btw_summary_perfs
        self.frame_skip = frame_skip

        self.max_start_nullops = max_start_nullops
        self.rng = rng

    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """
        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)

            if self.test_length > 0:
                print "starting testing: epoch, self.test_length"
                self.agent.start_testing()
                print epoch, self.test_length
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)
                print "end testing"
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
            
        print "self.agent.network.discount: "+str(self.agent.network.discount)
        print "self.agent.network.lr: "+str(self.agent.network.lr)
        print "self.agent.epsilon: "+str(self.agent.epsilon)

        steps_left = num_steps
        while steps_left > 0:
            print "num_epoch:"+str(epoch)
            _, num_steps = self.run_episode(steps_left, testing)
            steps_left -= num_steps

    def _init_episode(self, testing):
        """ This method resets the game, performs enough null
        actions to ensure that the buffer (the dataset) is ready
        """

        observation=self.environment.init(testing)
                
        self.agent.start_episode(observation)

        if self.max_start_nullops > 0:
            for _ in range(self.max_start_nullops):
                action=0#, V=self.agent.step(observation)
                self.agent.add_sample_1(observation)                
                reward, observation, terminal=self.environment.act(action, testing) # Null action
                self.agent.add_sample_2(reward)
                
        return reward, observation, terminal

    def _step(self, action, testing):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        
        reward = 0
        for _ in range(self.frame_skip):
            reward_, observation, terminal = self.environment.act(action, testing)
            reward += reward_

        return reward, observation, terminal

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        reward, observation, terminal = self._init_episode(testing)
        V_set=[]
        num_steps = 0
        while True:
                                    
            self.agent.add_sample_1(observation) ## NB : even if testing, it will just be a different queue
                        
            action, V = self.agent.step(observation)            
                        
            reward, observation, terminal = self._step(action, testing)
            
            #print "run_episode:action --> reward, observation, terminal"
            #print str(action)+" --> "+str(reward)+", "+str(observation)+", "+str(terminal)
                

            self.agent.add_sample_2(reward) ## NB : even if testing, it will just be a different queue

            if not testing:
                self.agent.do_training()
            
            V_set.append(V)
            num_steps += 1
            
            if terminal or num_steps >= max_steps:
                if not testing:
                    print "np.average(self.agent.loss_averages)"
                    #print self.agent.loss_averages
                    print np.average(self.agent.loss_averages)
                    
                    print np.average(V_set)
                
                break

        return terminal, num_steps
