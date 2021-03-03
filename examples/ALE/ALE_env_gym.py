""" Interface with the ALE environment

Authors: Vincent Francois-Lavet
"""
import numpy as np
np.set_printoptions(threshold=np.nan)
import cv2
#from ale_python_interface import ALEInterface
import gym
from deer.base_classes import Environment

#import matplotlib
#matplotlib.use('qt5agg')
#from mpl_toolkits.axes_grid1 import host_subplot
#import mpl_toolkits.axisartist as AA
#import matplotlib.pyplot as plt
#from PIL import Image
    
class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        if(bool(kwargs["game"])):
            self.env = gym.make(kwargs["game"])
        else:
            # Choice between Seaquest-v4, Breakout-v4, SpaceInvaders-v4, BeamRider-v4, Qbert-v4, Freeway-v4', etc.
            self.env = gym.make('Seaquest-v4')
        self._random_state=rng
        self.env.reset()
        frame_skip=kwargs.get('frame_skip',1)
        self._frame_skip = frame_skip if frame_skip >= 1 else 1
        
        self._screen=np.average(self.env.render(mode='rgb_array'),axis=-1)
        self._reduced_screen = cv2.resize(self._screen, (84, 84), interpolation=cv2.INTER_LINEAR) 
            #decide whether you want to keep this in repo, if so: add dependency to cv2
        #plt.imshow(self._reduced_screen, cmap='gray')
        #plt.show()
        
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0


                
    def reset(self, mode):
        if mode == self._mode:
            # already in the right mode
            self._mode_episode_count += 1
        else:
            # switching mode
            self._mode = mode
            self._mode_score = 0.0
            self._mode_episode_count = 0

        self.env.reset()
        for _ in range(self._random_state.randint(15)):
            action = self.env.action_space.sample()

            # this executes the environment with an action,
            # and returns the observation of the environment,
            # the reward, if the env is over, and other info.
            observation, reward, self.terminal, info = self.env.step(action)

        self._screen=np.average(self.env.render(mode='rgb_array'),axis=-1)
        self._reduced_screen = cv2.resize(self._screen, (84, 84), interpolation=cv2.INTER_LINEAR) 
        self.state=np.zeros((84,84), dtype=np.uint8) #FIXME
        
        return [1*[4 * [84 * [84 * [0]]]]]
        
        
    def act(self, action):
        #print "action"
        #print action
        
        self.state=np.zeros((4,84,84), dtype=np.float)
        reward=0
        for t in range(4):
            observation, r, self.terminal, info = self.env.step(action)
            #print "observation, reward, self.terminal"
            #print observation, reward, self.terminal
            reward+=r
            if self.inTerminalState():
                break

            self._screen=np.average(observation,axis=-1) # Gray levels
            self._reduced_screen = cv2.resize(self._screen, (84, 84), interpolation=cv2.INTER_NEAREST)  # 84*84
            #plt.imshow(self._screen, cmap='gray')
            #plt.show()
            self.state[t,:,:]=self._reduced_screen
            
        self._mode_score += reward
        return np.sign(reward)

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / self._mode_episode_count, self._mode_episode_count))


    def inputDimensions(self):
        return [(1, 4, 84, 84)]

    def observationType(self, subject):
        return np.float16

    def nActions(self):
        print ("self.env.action_space")
        print (self.env.action_space)
        return self.env.action_space.n

    def observe(self):
        return [(np.array(self.state)-128.)/128.]

    def inTerminalState(self):
        return self.terminal
                


if __name__ == "__main__":
    pass