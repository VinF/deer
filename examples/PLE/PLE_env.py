""" Interface with the PLE environment

Authors: Vincent Francois-Lavet, David Taralla
Modified by: Norman Tasfi
"""
import numpy as np
import cv2
from ple import PLE

from deer.base_classes import Environment

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, game=None, frame_skip=2,
            ple_options={"display_screen": True, "force_fps":True, "fps":30}):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._frame_skip = frame_skip if frame_skip >= 1 else 1
        self._random_state = rng
       
        if game is None:
            raise ValueError("Game must be provided")

        self._ple = PLE(game, **ple_options)
        self._ple.init()

        w, h = self._ple.getScreenDims()
        self._screen = np.empty((h, w), dtype=np.uint8)
        self._reduced_screen = np.empty((48, 48), dtype=np.uint8)
        self._actions = self._ple.getActionSet()

                
    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
            else:
                self._mode_episode_count += 1
        elif self._mode != -1: # and thus mode == -1
            self._mode = -1

        self._ple.reset_game()
        for _ in range(self._random_state.randint(15)):
            self._ple.act(self._ple.NOOP)
        self._screen = self._ple.getScreenGrayscale()
        cv2.resize(self._screen, (48, 48), self._reduced_screen, interpolation=cv2.INTER_NEAREST)
        
        return [2 * [48 * [48 * [0]]]]
        
        
    def act(self, action):
        action = self._actions[action]
        
        reward = 0
        for _ in range(self._frame_skip):
            reward += self._ple.act(action)
            if self.inTerminalState():
                break
            
        self._screen = self._ple.getScreenGrayscale()
        cv2.resize(self._screen, (48, 48), self._reduced_screen, interpolation=cv2.INTER_NEAREST)
  
        self._mode_score += reward
        return np.sign(reward)

    def summarizePerformance(self, test_data_set):
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / self._mode_episode_count, self._mode_episode_count))


    def inputDimensions(self):
        return [(2, 48, 48)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        return [np.array(self._reduced_screen)/256.]

    def inTerminalState(self):
        return self._ple.game_over()
                


if __name__ == "__main__":
    pass
