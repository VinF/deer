""" 
The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction.
Two actions are possible for the agent:
- Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
- Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.
The state of the agent is made up of an history of two punctual observations:
- The price signal
- Either the agent possesses the good or not (1 or 0)
The price signal is build following the same rules for the training and the validation environment. That allows the agent to learn a strategy that exploits this successfully.

Authors: Vincent Francois-Lavet, David Taralla
"""

import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import theano
import copy

from deer.base_classes import Environment

class MyEnv(Environment):
    
    def __init__(self, rng):
        """ Initialize environment.

        Parameters
        -----------
            rng : the numpy random number generator
        """
        # Defining the type of environment
        self._last_ponctual_observation = [0, 0] # At each time step, the observation is made up of two elements, each scalar
        
        self._random_state = rng
                
        # Building a price signal with some patterns
        self._price_signal=[]
        for i in range (1000):
            price = np.array([0.,0.,0.,-1.,0.,1.,0., 0., 0.])
            price += self._random_state.uniform(0, 3)
            self._price_signal.extend(price.tolist())
       
        self._price_signal_train = self._price_signal[:len(self._price_signal)//2]
        self._price_signal_valid = self._price_signal[len(self._price_signal)//2:]
        self._prices = None
        self._counter = 1
                
    def reset(self, mode):
        if mode == -1:
            self.prices = self._price_signal_train
        else:
            self.prices = self._price_signal_valid
            
        
        self._last_ponctual_observation = [self.prices[0], 0]

        self._counter = 1
        return [6*[0], 0]

    def act(self, action):
        reward = 0
        
        if (action == 0 and self._last_ponctual_observation[1] == 1):
            reward = self.prices[self._counter-1] - 0.5
        if (action == 1 and self._last_ponctual_observation[1] == 0):
            reward = -self.prices[self._counter-1] - 0.5

        self._last_ponctual_observation[0] = self.prices[self._counter]
        self._last_ponctual_observation[1] = action

        self._counter += 1
        
        return reward

    def summarizePerformance(self, test_data_set):
        """
        This function is called at every PERIOD_BTW_SUMMARY_PERFS.
        Parameters
        -----------
            test_data_set
        """
    
        print ("Summary Perf")
        
        observations = test_data_set.observations()
        prices = observations[0][100:200]
        invest = observations[1][100:200]
        
        steps=np.arange(len(prices))
        steps_long=np.arange(len(prices)*10)/10.
        
        #print steps,invest,prices
        host = host_subplot(111, axes_class=AA.Axes)
        plt.subplots_adjust(right=0.9, left=0.1)
    
        par1 = host.twinx()
    
        host.set_xlabel("Time")
        host.set_ylabel("Price")
        par1.set_ylabel("Investment")
    
        p1, = host.plot(steps_long, np.repeat(prices,10), lw=3, c = 'b', alpha=0.8, ls='-', label = 'Price')
        p2, = par1.plot(steps, invest, marker='o', lw=3, c = 'g', alpha=0.5, ls='-', label = 'Investment')
    
        par1.set_ylim(-0.09, 1.09)
    
    
        host.axis["left"].label.set_color(p1.get_color())
        par1.axis["right"].label.set_color(p2.get_color())
    
        plt.savefig("plot.png")
        print ("A plot of the policy obtained has been saved under the name plot.png")
    
    def inputDimensions(self):
        return [(6,), (1,)]     # We consider an observation made up of an history of 
                                # - the last six for the first scalar element obtained
                                # - the last one for the second scalar element


    def nActions(self):
        return 2                # The environment allows two different actions to be taken at each time step


    def inTerminalState(self):
        return False

    def observe(self):
        return np.array(self._last_ponctual_observation)

                


def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv = MyEnv(rng)

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
