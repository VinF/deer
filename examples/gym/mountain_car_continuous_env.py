""" Mountain car environment with continuous action space.

Author: Vincent Francois-Lavet
"""

import numpy as np
import copy
import math
from deer.base_classes import Environment
import gym

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        self.env = gym.make('MountainCarContinuous-v0')
        self.rng=rng
        self._last_observation = self.env.reset()
        self.is_terminal=False
        self._input_dim = [(1,), (1,)]
        
    def act(self, action):
        """ Simulate one time step in the environment.
        """
        reward=0

        for _ in range(5): # Increase the duration of one time step by a factor 5
            self._last_observation, r, self.is_terminal, info = self.env.step(action)
            reward+=r
            if(self.is_terminal==True):
                break
                
            if (self.mode==0): # Show the policy only at test time
                try:
                    self.env.render()
                except:
                    pass
                
        return reward/100. #Scale the reward so that it's 1 at maximum
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
        Mode : int
            -1 corresponds to training and 0 to test
        """
        self.mode=mode
        
        self._last_observation = self.env.reset()

        self.is_terminal=False
        

        return self._last_observation
                
    def inTerminalState(self):
        """ Tell whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).
        """
        return self.is_terminal

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        return [[self.env.action_space.low[0],self.env.action_space.high[0]]]

    def observe(self):
        return copy.deepcopy(self._last_observation)
        
def main():
    # This function can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv=MyEnv(rng)
    print(myenv.env.action_space)
    print(myenv.env.action_space.low)
    print(myenv.env.action_space.high)    
    print(myenv.env.observation_space)
    
    print (myenv.observe())
    myenv.act([0])
    print (myenv.observe())
    myenv.act([1])
    print (myenv.observe())
    
    
if __name__ == "__main__":
    main()
