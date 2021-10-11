import copy
import math
import sys

import gym
import numpy as np

from deer.base_classes import Environment


class MyEnv(Environment):
    def __init__(self, rng):
        """Initialize environment.

        Arguments:
            rng - the numpy random number generator
        """
        gym.envs.register(
            id="MountainCarModified-v0",
            entry_point="gym.envs.classic_control:MountainCarEnv",
            max_episode_steps=500,  # MountainCar-v0 uses 200
            reward_threshold=-110.0,
        )

        self.env = gym.make("MountainCarModified-v0")
        self.env.max_episode_steps = 500
        self.rng = rng
        self._last_observation = self.env.reset()
        self.is_terminal = False
        self._input_dim = [(1,), (1,)]  # self.env.observation_space.shape is equal to 2
        # and we use only the current observation in the pseudo-state

    def act(self, action):
        """Simulate one time step in the environment."""
        reward = 0
        nsteps = 10
        for _ in range(nsteps):
            self._last_observation, r, self.is_terminal, info = self.env.step(action)
            reward += r
            if self.is_terminal == True:
                reward += 3 * nsteps
                break

            if self.mode == 0:  # Show the policy only at test time
                try:
                    self.env.render()
                except:
                    pass
                    # print("Warning:", sys.exc_info()[0])

        # s=copy.deepcopy(self._last_observation)
        ## Possibility to add a reward shaping for faster convergence
        # s[0]+=math.pi/6
        # if(s[0]>0):
        #    reward+=pow(s[0],2)#np.linalg.norm(s[0])

        return reward / nsteps

    def reset(self, mode=0):
        """Reset environment for a new episode.

        Arguments:
        Mode : int
            -1 corresponds to training and 0 to test
        """
        self.mode = mode

        self._last_observation = self.env.reset()
        # DEEPRECATED
        # if (self.mode==-1): # Reset to a random value when in training mode (that allows to increase exploration)
        #    high=self.env.observation_space.high
        #    low=self.env.observation_space.low
        #    self._last_observation=low+self.rng.rand(2)*(high-low)
        #    self.env.env.state=self._last_observation

        self.is_terminal = False

        return self._last_observation

    def inTerminalState(self):
        """Tell whether the environment reached a terminal state after the last transition (i.e. the last transition
        that occured was terminal).
        """
        return self.is_terminal

    def inputDimensions(self):
        return self._input_dim

    def nActions(self):
        return 3  # Would be useful to have this directly in gym : self.env.action_space.shape

    def observe(self):
        return copy.deepcopy(self._last_observation)


def main():
    # This function can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print(myenv.observe())


if __name__ == "__main__":
    main()
