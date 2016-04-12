""" The environment simulates the behavior of an inverted pendulum.
The goal of the agent, as suggested by the reward function, is 
to balance a pole on a cart that can either move left or right.

Please refer to the wiki for a complete decription of the problem.

Author: Aaron Zixiao Qiu
"""

import numpy as np
import copy

import theano

from render_movie import save_mp4
from deeprl.base_classes import Environment

# Physics constants
G = 9.8 
M_CART = 1.0
M_POLE = 0.1
L = 0.5
F = 10
DELTA_T = 0.02
PI = np.pi

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        # Defining the type of environment
        self._rng = rng
        # Observations = (x, x_dot, theta, theta_dot, timestamp)
        self._last_observation = [0, 0, self._rng.uniform(-PI/2, PI/2), 0, 0]
        self._input_dim = [(1,), (1,), (1,), (1,), (1,)]
        self._video = 0

    def act(self, action):
        """ This is the most important function in the environment. 
        We simulate one time step in the environment. Given an input 
        action, compute the next state of the system (position, speed, 
        angle, angular speed) and return a reward. 
        
        Argument:
            action - 0: move left (F = -10N); 1: move right (F = +10N)
        Return:
            reward - reward for this transition
        """
        # Direction of the force
        force = F
        if (action == 0):
            force = -F

        # The cart cannot move beyond -1 or 1
        if (self._last_observation[0] <= -1 and force < 0):
            force = 0
        elif (self._last_observation[0] >= 1 and force > 0):
            force = 0

        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = 10
        tau = DELTA_T / n_tau
        for i in range(n_tau):
            # Physics -> See wiki for the formulas
            x, x_dot, theta, theta_dot, _ = self._last_observation
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            tmp = (force + M_POLE*L*sin_theta*theta_dot**2) / (M_POLE + M_CART)
            theta_dd = (G*sin_theta - cos_theta*tmp) / \
                       (L*(4/3. - M_POLE*cos_theta**2/(M_POLE + M_CART)))
            x_dd = tmp - M_POLE*theta_dd*cos_theta/(M_POLE + M_CART)

            # Update observation vector
            self._last_observation = [
                x + tau*x_dot,
                x_dot + tau*x_dd,
                self._to_range(theta + tau*theta_dot),
                theta_dot + tau*theta_dd,
                self._last_observation[4] + tau
                ]

            # As mentionned, the cart cannot move beyond -1 or 1
            if (self._last_observation[0] < -1):
                self._last_observation[0] = -1
                self._last_observation[1] = max([0, self._last_observation[1]])
            elif (self._last_observation[0] > 1):
                self._last_observation[0] = 1
                self._last_observation[1] = min([0, self._last_observation[1]])

        # Simple reward
        theta = self._last_observation[2]
        reward = -abs(theta) 

        # Penalize the reward with respect to the position x
        reward -= abs(self._last_observation[0])

        return reward
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
            mode - Not used in this example.
        """
        # Reset initial observation to a random theta
        self._last_observation = [0, 0, self._rng.uniform(-PI/2, PI/2), 0, 0]

        return self._last_observation
        
    def summarizePerformance(self, test_data_set):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            test_data_set - Simulation data returned by the agent.
        """
        print ("Summary Perf")
        if (self._video < 1):
            self._video += 1
            return

        # Save the data in the correct input format for video generation
        observations = test_data_set.observations()
        data = np.zeros((len(observations[0]), len(observations)))
        for i in range(1, 5):
            data[:,i] = observations[i - 1]
        data[:,0] = observations[4]
        
        save_mp4(data, self._video)
        self._video += 1
        return
    
    def _to_range(self, angle):
        # Convert theta in the range [-PI, PI]
        n = abs(angle) // (2*PI)
        if (angle < 0):
            angle += n*2*PI
        else:
            angle -= n*2*PI

        if (angle < -PI):
            angle = 2*PI - abs(angle)
        elif (angle > PI):
            angle = -(2*PI - angle)

        return angle

    def inputDimensions(self):
        return self._input_dim  

    def nActions(self):
        # The environment allows two different actions to be taken
        # at each time step
        return 2                   

    def observe(self):
        return copy.deepcopy(self._last_observation)  

