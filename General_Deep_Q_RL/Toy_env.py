import numpy as np

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

class Env(object):
    def __init__(self, rng):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator            
        """
        self.rng = rng

        # Defining the type of environment
        self.observation=[0,0] # At each time step, the observation is made up of two elements, each scalar
        self.num_actions=2 # The environment allows two different actions to be taken at each time step
        self.num_elements_in_batch=[6,1]  # We consider a belief state made up of an history of 
                                          # - the last six for the first element obtained 
                                          # - the last one for the second element
                
        # Building a price signal with some patterns
        self.price_signal=[]
        for i in range (1000):
            price=np.array([0.,0.,0.,-1.,0.,1.,0., 0., 0.])
            price+=self.rng.uniform(0, 3)
            
            self.price_signal.extend(price.tolist())
       
        self.price_signal_train=self.price_signal[:len(self.price_signal)/2]
        self.price_signal_valid=self.price_signal[len(self.price_signal)/2:]


        
    def init(self, testing):
        """ Reset environment for a new episode
        Arguments:
            testing - whether we are in test mode or train mode (boolean)  
        Returns:
            self.observation - current observation (list of k elements)
        """
        if(testing):
            self.prices=self.price_signal_valid
        else:
            self.prices=self.price_signal_train
            
        
        self.observation=[self.prices[0],0]
        
        self.counter = 1     

        return self.observation
        
        
    def act(self, action, testing):
        """
        Performs one time step on the environment
        Arguments:
            action - chosen action (integer)
            testing - whether we are in test mode or train mode (boolean)  
        Returns:
           reward - obtained reward for this transition
           self.observation - new observation
           terminal - whether this is the end of an episode (boolean)
        """
        #print "NEW STEP"

        reward = 0
        terminal=0
        
        if (action==0 and self.observation[1]==1):
            reward=self.prices[self.counter-1]
            reward-=0.5    
        if (action==1 and self.observation[1]==0):
            reward=-self.prices[self.counter-1]        
            reward-=0.5    

        if (action==0):
            self.observation[1]=0
        if (action==1):
            self.observation[1]=1

        self.observation[0]=self.prices[self.counter]
                    

        self.counter +=1

        return reward, self.observation, terminal





    def get_summary_perf(self, test_data_set):
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


        
                


def main():
    # Can be used for debug purposes
    rng = np.random.RandomState(123456)
    myenv=Env(rng)
    myenv.init(0)
    
    print myenv.act(1, False)
    print myenv.act(1, False)
    print myenv.act(0, False)
    print myenv.act(0, False)
    print myenv.act(1, False)
    
if __name__ == "__main__":
    main()
