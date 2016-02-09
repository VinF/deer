import numpy as np
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import theano
from environment import Environment

class MyEnv(Environment):
    def __init__(self, rng):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator            
        """
        self._randomState = rng

        # Defining the type of environment
        self._lastPonctualObservation = [0, 0] # At each time step, the observation is made up of two elements, each scalar
        self._nActions = 2                     # The environment allows two different actions to be taken at each time step
        self._batchDimensions = [(6,), (1,)] # We consider a belief state made up of an history of 
                                             # - the last six for the first element obtained 
                                             # - the last one for the second element
        self._state = []
        for i in range(len(self._batchDimensions)):
            dim = self._batchDimensions[i] + np.array(self._lastPonctualObservation[i]).shape
            self._state.append(np.zeros(dim, dtype=theano.config.floatX))
                
        # Building a price signal with some patterns
        self._priceSignal=[]
        for i in range (1000):
            price = np.array([0.,0.,0.,-1.,0.,1.,0., 0., 0.])
            price += self._randomState.uniform(0, 3)
            self._priceSignal.extend(price.tolist())
       
        self._priceSignalTrain = self._priceSignal[:len(self._priceSignal)/2]
        self._priceSignalValid = self._priceSignal[len(self._priceSignal)/2:]
        self._prices = None
        self._counter = 1
                
    def reset(self, testing):
        """ Reset environment for a new episode.

        Arguments:
            testing - whether we are in test mode or train mode (boolean)  
        """
        if testing:
            self.prices = self._priceSignalValid
        else:
            self.prices = self._priceSignalTrain
            
        
        self._lastPonctualObservation = [np.array([self.prices[0], 0, 0]), 0]
        for i in range(len(self._lastPonctualObservation)):
            self._state[i] = np.zeros_like(self._state[i])
            self._state[i][-1] = self._lastPonctualObservation[i]

        
        self._counter = 1
        
        
    def act(self, action, testing):
        """
        Performs one time step on the environment
        Arguments:
            action - chosen action (integer)
            testing - whether we are in test mode or train mode (boolean)  
        Returns:
           reward - obtained reward for this transition
        """
        reward = 0
        
        if (action == 0 and self._lastPonctualObservation[1] == 1):
            reward = self.prices[self._counter-1] - 0.5
        if (action == 1 and self._lastPonctualObservation[1] == 0):
            reward = -self.prices[self._counter-1] - 0.5

        self._lastPonctualObservation[0] = [self.prices[self._counter], action, action]
        self._lastPonctualObservation[1] = action        
        for i in range(len(self._lastPonctualObservation)):
            if (self._state[i].ndim == 2):
                self._state[i] = np.roll(self._state[i], -1, axis=0)
            else:
                self._state[i] = np.roll(self._state[i], -1)
            self._state[i][-1] = self._lastPonctualObservation[i]

        self._counter +=1

        return reward



    def summarizePerformance(self, test_data_set):
        """
        This function is called at every PERIOD_BTW_SUMMARY_PERFS.
        Arguments:
            test_data_set
        """
    
        print "Summary Perf"
        
        prices=[]
        invest=[]
        for elems in test_data_set.elements[100:125]:
            prices.append(elems[0])
            invest.append(elems[1])
        
        prices=np.array(prices)
        invest=np.array(invest)
        steps=np.arange(25)
        steps_long=np.arange(250)/10.
        
        print steps,invest,prices
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
    
        plt.draw()
        plt.show()

    def batchDimensions(self):
        return self._batchDimensions

    def inTerminalState(self):
        return False

    def isSuccess(self):
        return True

    def observe(self):
        return np.array(self._lastPonctualObservation)

    def state(self):
        return self._state

                


def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv=Env(rng)
    myenv.reset(False)
    
    myenv.act(1, False)
    myenv.act(1, False)
    myenv.act(0, False)
    myenv.act(0, False)
    myenv.act(1, False)

    print myenv._state
    
if __name__ == "__main__":
    main()
