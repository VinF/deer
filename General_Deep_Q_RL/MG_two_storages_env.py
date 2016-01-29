import numpy as np
import MG_data
np.set_printoptions(threshold=np.nan)

from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

class Env(object):
    def __init__(self, rng):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator
            
        """

        self.consumption, self.min_consumption, self.max_consumption=MG_data.get_consumption(365*24)
        print "self.consumption: " + str(self.consumption[0:100])
        print self.min_consumption, self.max_consumption, self.consumption.shape

        self.production_train, self.min_production, self.max_production=MG_data.get_production(0,365*24)
        self.production_valid, self.min_production_valid, self.max_production_valid=MG_data.get_production(365*24,365*24)
        self.production_train=self.production_train*16000/1000 #(16KWp (80m^2) et en kWh)
        print "self.production_train: " + str(self.production_train[0:100])
        print self.production_train.shape

        self.rng = rng

        self.battery_size=20.
        self.battery_eta=0.9
        
        self.hydrogen_max_power=0.5
        self.hydrogen_eta=.65        
        
        self.observation=[0. ,0.,0.,0.,0.,0.,0.]
        self.num_actions=3
        self.num_elements_in_batch=[1, 1,12,1,12,1,1] # [battery storage,   consumption_short, consumption_long, production_short, production_long]
        
        
    def init(self, testing):
        """
        Returns:
           current observation (list of k elements)
        """
        ### Test 6
        self.observation=[1., 0.,0.,0.,0.,0.,0.]
        self.counter = 1        
        self.hydrogen_storage=0.

        return self.observation
        
        
    def act(self, action, testing):
        """
        Perform one time step on the environment
        """
        #print "NEW STEP"

        reward = 0#self.ale.act(action)  #FIXME
        terminal=0


        ### Test
        # Case with one battery, a variable grid price, an internal consumption
        # self.observation[0] : battery state (normalised [0,1])
        # self.observation[1] : consumption_short (normalised [-1,1])
        # self.observation[2] : consumption_short (normalised [-1,1])
        # self.observation[3] : production_short (normalised [-1,1])
        # self.observation[4] : production_long (normalised [-1,1])
        ###

        #true_market_price=self.market_price[self.counter-1]*(self.max_market_price-self.min_market_price)+self.min_market_price
        true_cons=self.consumption[self.counter]*(self.max_consumption-self.min_consumption)+self.min_consumption
        true_prod=self.production_train[self.counter]#*(self.max_production-self.min_production)+self.min_production
        
        true_demand=true_cons-true_prod
        
        #print "action:"+str(action)
        #print "state b"+str(self.observation[0])
        #print "true_demand"+str(true_demand)
        if (action==0):
            ## Energy is taken out of the hydrogen reserve
            true_energy_avail_from_hydrogen=-self.hydrogen_max_power
            diff_hydrogen=-self.hydrogen_max_power/self.hydrogen_eta
        if (action==1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen=0
            diff_hydrogen=0
        if (action==2):
            ## Energy is taken into the hydrogen reserve
            true_energy_avail_from_hydrogen=self.hydrogen_max_power
            diff_hydrogen=self.hydrogen_max_power*self.hydrogen_eta
            
        reward=diff_hydrogen*0.1 # 0.1euro/kWh of hydrogen
        self.hydrogen_storage+=diff_hydrogen

        Energy_needed_from_battery=true_demand+true_energy_avail_from_hydrogen
        
        #print "self.observation[0],true_demand, action, Energy_needed_from_battery"
        #print self.observation[0],true_demand, action, Energy_needed_from_battery

        if (Energy_needed_from_battery>0):
        # Lack of energy
            if (self.observation[0]*self.battery_size>Energy_needed_from_battery):
            # If enough energy in the battery, use it
                self.observation[0]=self.observation[0]-Energy_needed_from_battery/self.battery_size
            else:
            # Otherwise: use what is left and then penalty                
                reward=-(Energy_needed_from_battery-self.observation[0]*self.battery_size)*2 #2euro/kWh
                self.observation[0]=0
        if (Energy_needed_from_battery<0):
        # Surplus of energy --> load the battery
            self.observation[0]=min(1.,self.observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)
                    
        
        #print "reward,self.observation[0]"
        #print reward,self.observation[0]
            
        

        self.observation[1]=self.consumption[self.counter]
        self.observation[2]=self.consumption[self.counter]
        self.observation[3]=self.production_train[self.counter]
        self.observation[4]=self.production_train[self.counter]
        self.observation[5]=sum(self.production_train[self.counter:self.counter+24])/24.
        self.observation[6]=sum(self.production_train[self.counter:self.counter+48])/48.

                    
        self.counter+=1
                
        return reward, self.observation, terminal

        

    def get_summary_perf(self, test_data_set):
        print "summary perf"
        print "self.hydrogen_storage: "+str(self.hydrogen_storage)
        
        for i in range(1):
            battery_level=[]
            production=[]
            consumption=[]
            for elems in test_data_set.elements[i+0:i+100]:
                battery_level.append(elems[0])
                consumption.append(elems[1])
                production.append(elems[3])
            
            actions=test_data_set.actions[i+0:i+100]
            
            
            battery_level=np.array(battery_level)*self.battery_size
            consumption=np.array(consumption)*(self.max_consumption-self.min_consumption)+self.min_consumption
            production=np.array(production)#*(self.max_production-self.min_production)+self.min_production

            steps=np.arange(100)
            print steps
            print production
            steps_long=np.arange(1000)/10.
            
            
            host = host_subplot(111, axes_class=AA.Axes)
            plt.subplots_adjust(left=0.2, right=0.8)
            
            par1 = host.twinx()
            par2 = host.twinx()
            par3 = host.twinx()
            
            offset = 60
            new_fixed_axis = par2.get_grid_helper().new_fixed_axis
            par2.axis["right"] = new_fixed_axis(loc="right",
                                                axes=par2,
                                                offset=(offset, 0))    
            par2.axis["right"].toggle(all=True)
            
            offset = -60
            new_fixed_axis = par3.get_grid_helper().new_fixed_axis
            par3.axis["right"] = new_fixed_axis(loc="left",
                                                axes=par3,
                                                offset=(offset, 0))    
            par3.axis["right"].toggle(all=True)
            
            
            host.set_xlim(-0.9, 99)
            host.set_ylim(0, 15.9)
            
            host.set_xlabel("Time")
            host.set_ylabel("Battery level")
            par1.set_ylabel("Consumption")
            par2.set_ylabel("Production")
            par3.set_ylabel("H Actions")
            
            p1, = host.plot(steps, battery_level, marker='o', lw=1, c = 'b', alpha=0.8, ls='-', label = 'Battery level')
            p2, = par1.plot(steps_long-0.9, np.repeat(consumption,10), lw=3, c = 'r', alpha=0.5, ls='-', label = 'Consumption')
            p3, = par2.plot(steps_long-0.9, np.repeat(production,10), lw=3, c = 'g', alpha=0.5, ls='-', label = 'Production')
            p4, = par3.plot(steps_long, np.repeat(actions,10), lw=3, c = 'c', alpha=0.5, ls='-', label = 'H Actions')
            
            par1.set_ylim(0, 10.09)
            par2.set_ylim(0, 10.09)
            par3.set_ylim(-0.09, 2.09)
            
            #host.legend(loc=2)#loc=9)
            
            host.axis["left"].label.set_color(p1.get_color())
            par1.axis["right"].label.set_color(p2.get_color())
            par2.axis["right"].label.set_color(p3.get_color())
            par3.axis["right"].label.set_color(p4.get_color())
            
            plt.savefig("plot.png")
            
            plt.draw()
            plt.show()
            plt.close('all')



def main():
    rng = np.random.RandomState(123456)
    myenv=Env(rng)

    print "market price"

    #aa, minaa,maxaa=get_market_price()
    #print aa[0:100], minaa,maxaa    

    print "consumption"
    aa, minaa,maxaa=MG_data.get_consumption(100)
    print aa[0:100], minaa,maxaa
    
    print "production"
    aa, minaa,maxaa=MG_data.get_production(0,100)
    aa, minaa,maxaa=MG_data.get_production(365*24,100)
    print aa[0:100], minaa,maxaa
    
    
    print myenv.act(1, False)
    print myenv.act(1, False)
    print myenv.act(0, False)
    print myenv.act(0, False)
    
if __name__ == "__main__":
    main()
