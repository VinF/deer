""" The environment simulates the behavior of an inverted pendulum.
The goal of the agent, as suggested by the reward function, is 
to balance a pole on a cart that can either move left or right.

Code is based on the following inverted pendulum implementations
in C : http://webdocs.cs.ualberta.ca/%7Esutton/book/code/pole.c
in Python : https://github.com/toddsifleet/inverted_pendulum

Please refer to the wiki for a complete decription of the problem.

Author: Aaron Zixiao Qiu
"""

import numpy as np
import copy

import theano

from render_movie import save_mp4
from deer.base_classes import Environment

# Physics constants
G = 9.8 
M_CART = 1.0
M_POLE = 0.1
L = 0.5
F = 100
DELTA_T = 0.02
PI = np.pi
MU_C = 0.0005
MU_P = 0.000002

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment.

        Arguments:
            rng - the numpy random number generator            
        """
        # Defining the type of environment
        self._rng = rng
        # Observations = (x, x_dot, theta, theta_dot, timestamp)
        self._last_observation = [0, 0, 0, 0]
        self._input_dim = [(1,), (1,), (1,), (1,)]
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

        # Divide DELTA_T into smaller tau's, to better take into account
        # the transitions
        n_tau = 10
        tau = DELTA_T / n_tau
        for i in range(n_tau):
            # Physics -> See wiki for the formulas
            x, x_dot, theta, theta_dot, = self._last_observation#_ = self._last_observation
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)

            f_cart = MU_C * np.sign(x_dot)
            f_pole = MU_P * theta_dot / (M_POLE*L)

            tmp = (force + M_POLE*L*sin_theta*theta_dot**2 - f_cart) \
                  / (M_POLE + M_CART)
            theta_dd = (G*sin_theta - cos_theta*tmp - f_pole) \
                       / (L*(4/3. - M_POLE*cos_theta**2/(M_POLE + M_CART))) 
            x_dd = tmp - M_POLE*theta_dd*cos_theta/(M_POLE + M_CART)

            # Update observation vector
            self._last_observation = [
                x + tau*x_dot,
                x_dot + tau*x_dd,
                self._to_range(theta + tau*theta_dot),
                theta_dot + tau*theta_dd,
                ]

        # Simple reward
        reward = - abs(theta) 
        reward -= abs(self._last_observation[0])/2.

        # The cart cannot move beyond -5 or 5
        if(self._last_observation[0]<-5):
            self._last_observation[0]=-5
        if(self._last_observation[0]>5):
            self._last_observation[0]=5
 

        return reward
                
    def reset(self, mode=0):
        """ Reset environment for a new episode.

        Arguments:
            mode - Not used in this example.
        """
        # Reset initial observation to a random x and theta
        x = self._rng.uniform(-1, 1)
        theta = self._rng.uniform(-PI, PI)
        self._last_observation = [x, 0, theta, 0]

        return self._last_observation
        
    def summarizePerformance(self, test_data_set):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            test_data_set - Simulation data returned by the agent.
        """
        print ("Summary Perf")

        # Save the data in the correct input format for video generation
        observations = test_data_set.observations()
        data = np.zeros((len(observations[0]), len(observations)))
        for i in range(1, 4):
            data[:,i] = observations[i - 1]
        data[:,0]=np.arange(len(observations[0]))*0.02
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

