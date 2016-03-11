import numpy as np
import cv2
from ale_python_interface import ALEInterface
from base_classes import Environment

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
from IPython import embed

class MyEnv(Environment):
    VALIDATION_MODE = 0
    TEST_MODE = 1

    def __init__(self, rng, rom="ale/breakout.bin", frame_skip=4, 
                 ale_options=[{"key": "random_seed", "value": 0}, {"key": "color_averaging", "value": True},
                              {"key": "repeat_action_probability", "value": 0.}]):
        self._mode = -1
        self._modeScore = 0
        self._modeEpisodeCount = 0

        self._frameSkip = frame_skip if frame_skip >= 1 else 1
        self._randomState = rng

        self._ale = ALEInterface()
        for option in ale_options:
            t = type(option["value"])
            if t is int:
                self._ale.setInt(option["key"], option["value"])
            elif t is float:
                self._ale.setFloat(option["key"], option["value"])
            elif t is bool:
                self._ale.setBool(option["key"], option["value"])
            else:
                raise ValueError("Option {} ({}) is not an int, bool or float.".format(option["key"], t))
        self._ale.loadROM(rom)

        w, h = self._ale.getScreenDims()
        self._screen = np.empty((h, w), dtype=np.uint8)
        self._reducedScreen = np.empty((84, 84), dtype=np.uint8)
        self._actions = self._ale.getMinimalActionSet()

                
    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._modeScore = 0
                self._modeEpisodeCount = 0
            else:
                self._modeEpisodeCount += 1
        elif self._mode != -1: # and thus mode == -1
            self._mode = -1

        self._ale.reset_game()
        for _ in range(self._randomState.randint(15)):
            self._ale.act(0)
        self._ale.getScreenGrayscale(self._screen)
        cv2.resize(self._screen, (84, 84), self._reducedScreen, interpolation=cv2.INTER_NEAREST)
        
        return [4 * [84 * [84 * [0]]]]
        
        
    def act(self, action):
        action = self._actions[action]
        
        reward = 0
        for _ in range(self._frameSkip):
            reward += self._ale.act(action)
            if self.inTerminalState():
                break
            
        self._ale.getScreenGrayscale(self._screen)
        cv2.resize(self._screen, (84, 84), self._reducedScreen, interpolation=cv2.INTER_NEAREST)
  
        self._modeScore += reward
        return np.sign(reward)

    def summarizePerformance(self, test_data_set):
        if self.inTerminalState() == False:
            self._modeEpisodeCount += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._modeScore / self._modeEpisodeCount, self._modeEpisodeCount))


    def inputDimensions(self):
        return [(4, 84, 84)]

    def nActions(self):
        return len(self._actions)

    def observe(self):
        return [np.array(self._reducedScreen)]

    def inTerminalState(self):
        return self._ale.game_over()
                


if __name__ == "__main__":
    pass