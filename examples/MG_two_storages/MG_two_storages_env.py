"""
The environment simulates a microgrid consisting of short and long term storage. The agent can either choose to store in the long term storage or take energy out of it while the short term storage handle at best the lack or surplus of energy by discharging itself or charging itself respectively. Whenever the short term storage is empty and cannot handle the net demand a penalty (negative reward) is obtained equal to the value of loss load set to 2euro/kWh.
Two actions are possible for the agent:
- Action 0 corresponds to discharging the long-term storage
- Action 1 corresponds to charging the long-term storage
The state of the agent is made up of an history of two to four punctual observations:
- Charging state of the short term storage (0 is empty, 1 is full)
- Production and consumption (0 is no production or consumption, 1 is maximal production or consumption)
( - Distance to equinox )
( - Predictions of future production : average of the production for the next 24 hours and 48 hours )
More information can be found in the paper to be published :
Efficient decision making in stochastic micro-grids using deep reinforcement learning, Vincent Francois-Lavet, David Taralla, Raphael Fonteneau, Damien Ernst

Authors: Vincent Francois-Lavet, David Taralla
"""

import numpy as np
np.set_printoptions(threshold=np.nan)
import copy

from deer.base_classes import Environment
from plot_MG_operation import plot_op

class MyEnv(Environment):
    VALIDATION_MODE = 0
    TEST_MODE = 1

    def __init__(self, rng):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator
        """
        # Defining the type of environment
        self._dist_equinox=0
        self._pred=0
        
        inc_sizing=1.
        
        if (self._dist_equinox==1 and self._pred==1):
            self._last_ponctual_observation = [0. ,[0.,0.],0., [0.,0.]]
            self._input_dimensions = [(1,), (12,2), (1,),(1,2)]
        elif (self._dist_equinox==1 and self._pred==0):
            self._last_ponctual_observation = [0. ,[0.,0.],0.]
            self._input_dimensions = [(1,), (12,2), (1,)]
        elif (self._dist_equinox==0 and self._pred==0):
            self._last_ponctual_observation = [0. ,[0.,0.]]
            self._input_dimensions = [(1,), (12,2)]

        self._rng = rng

        # Get consumption profile in [0,1]
        self.consumption_train_norm=np.load("data/example_nondeterminist_cons_train.npy")[0:1*365*24]
        self.consumption_valid_norm=np.load("data/example_nondeterminist_cons_train.npy")[365*24:2*365*24]
        self.consumption_test_norm=np.load("data/example_nondeterminist_cons_test.npy")[0:1*365*24]
        # Scale consumption profile in [0,2.1kW] --> average max per day = 1.7kW, average per day is 18.3kWh
        self.consumption_train=self.consumption_train_norm*2.1
        self.consumption_valid=self.consumption_valid_norm*2.1
        self.consumption_test=self.consumption_test_norm*2.1

        self.min_consumption=min(self.consumption_train)
        self.max_consumption=max(self.consumption_train)
        print("Sample of the consumption profile (kW): {}".format(self.consumption_train[0:24]))
        print("Min of the consumption profile (kW): {}".format(self.min_consumption))
        print("Max of the consumption profile (kW): {}".format(self.max_consumption))
        print("Average consumption per day train (kWh): {}".format(np.sum(self.consumption_train)/self.consumption_train.shape[0]*24))
        print("Average consumption per day valid (kWh): {}".format(np.sum(self.consumption_valid)/self.consumption_valid.shape[0]*24))
        print("Average consumption per day test (kWh): {}".format(np.sum(self.consumption_test)/self.consumption_test.shape[0]*24))

        # Get production profile in W/Wp in [0,1]
        self.production_train_norm=np.load("data/BelgiumPV_prod_train.npy")[0:1*365*24]
        self.production_valid_norm=np.load("data/BelgiumPV_prod_train.npy")[365*24:2*365*24] #determinist best is 110, "nondeterminist" is 124.9
        self.production_test_norm=np.load("data/BelgiumPV_prod_test.npy")[0:1*365*24] #determinist best is 76, "nondeterminist" is 75.2
        # Scale production profile : 12KWp (60m^2) et en kWh
        self.production_train=self.production_train_norm*12000./1000.*inc_sizing
        self.production_valid=self.production_valid_norm*12000./1000.*inc_sizing
        self.production_test=self.production_test_norm*12000/1000*inc_sizing

        self.min_production=min(self.production_train)
        self.max_production=max(self.production_train)
        print("Sample of the production profile (kW): {}".format(self.production_train[0:24]))
        print("Min of the production profile (kW): {}".format(self.min_production))
        print("Max of the production profile (kW): {}".format(self.max_production))
        print("Average production per day train (kWh): {}".format(np.sum(self.production_train)/self.production_train.shape[0]*24))
        print("Average production per day valid (kWh): {}".format(np.sum(self.production_valid)/self.production_valid.shape[0]*24))
        print("Average production per day test (kWh): {}".format(np.sum(self.production_test)/self.production_test.shape[0]*24))

        self.battery_size=15.*inc_sizing
        self.battery_eta=0.9
        
        self.hydrogen_max_power=1.1*inc_sizing
        self.hydrogen_eta=.65
        
    def reset(self, mode):
        """
        Returns:
           current observation (list of k elements)
        """
        ### Test 6
        if (self._dist_equinox==1 and self._pred==1):
            self._last_ponctual_observation = [1. ,[0.,0.],0., [0.,0.]]
        elif (self._dist_equinox==1 and self._pred==0):
            self._last_ponctual_observation = [1. ,[0.,0.],0.]
        elif (self._dist_equinox==0 and self._pred==0):
            self._last_ponctual_observation = [1. ,[0.,0.]]

        self.counter = 1        
        self.hydrogen_storage=0.
        
        if mode == -1:
            self.production_norm=self.production_train_norm
            self.production=self.production_train
            self.consumption_norm=self.consumption_train_norm
            self.consumption=self.consumption_train
        elif mode == MyEnv.VALIDATION_MODE:
            self.production_norm=self.production_valid_norm
            self.production=self.production_valid
            self.consumption_norm=self.consumption_valid_norm
            self.consumption=self.consumption_valid
        else:
            self.production_norm=self.production_test_norm
            self.production=self.production_test
            self.consumption_norm=self.consumption_test_norm
            self.consumption=self.consumption_test
            
        if (self._dist_equinox==1 and self._pred==1):
            return [
                        0., 
                        [[0. ,0.] for i in range(12)],
                        0.,
                        [0.,0.]
                    ]
        elif (self._dist_equinox==1 and self._pred==0):
            return [
                        0., 
                        [[0. ,0.] for i in range(12)],
                        0.
                    ]
        else: #elif (self._dist_equinox==0, self._pred==0):
            return [
                        0., 
                        [[0. ,0.] for i in range(12)],
                    ]
        
    def act(self, action):
        """
        Perform one time step on the environment
        """
        #print "NEW STEP"

        reward = 0#self.ale.act(action)  #FIXME
        terminal=0

        true_demand=self.consumption[self.counter-1]-self.production[self.counter-1]
        
        if (action==0):
            ## Energy is taken out of the hydrogen reserve
            true_energy_avail_from_hydrogen=-self.hydrogen_max_power*self.hydrogen_eta
            diff_hydrogen=-self.hydrogen_max_power
        if (action==1):
            ## No energy is taken out of/into the hydrogen reserve
            true_energy_avail_from_hydrogen=0
            diff_hydrogen=0
        if (action==2):
            ## Energy is taken into the hydrogen reserve
            true_energy_avail_from_hydrogen=self.hydrogen_max_power/self.hydrogen_eta
            diff_hydrogen=self.hydrogen_max_power
            
        reward=diff_hydrogen*0.1 # 0.1euro/kWh of hydrogen
        self.hydrogen_storage+=diff_hydrogen

        Energy_needed_from_battery=true_demand+true_energy_avail_from_hydrogen
        
        if (Energy_needed_from_battery>0):
        # Lack of energy
            if (self._last_ponctual_observation[0]*self.battery_size>Energy_needed_from_battery):
            # If enough energy in the battery, use it
                self._last_ponctual_observation[0]=self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size/self.battery_eta
            else:
            # Otherwise: use what is left and then penalty                
                reward-=(Energy_needed_from_battery-self._last_ponctual_observation[0]*self.battery_size)*2 #2euro/kWh
                self._last_ponctual_observation[0]=0
        elif (Energy_needed_from_battery<0):
        # Surplus of energy --> load the battery
            self._last_ponctual_observation[0]=min(1.,self._last_ponctual_observation[0]-Energy_needed_from_battery/self.battery_size*self.battery_eta)
                    
        #print "new self._last_ponctual_observation[0]"
        #print self._last_ponctual_observation[0]
        
        ### Test
        # self._last_ponctual_observation[0] : State of the battery (0=empty, 1=full)
        # self._last_ponctual_observation[1] : Normalized consumption at current time step (-> not available at decision time)
        # self._last_ponctual_observation[1][1] : Normalized production at current time step (-> not available at decision time)
        # self._last_ponctual_observation[2][0] : Prevision (accurate) for the current time step and the next 24hours
        # self._last_ponctual_observation[2][1] : Prevision (accurate) for the current time step and the next 48hours
        ###
        self._last_ponctual_observation[1][0]=self.consumption_norm[self.counter]
        self._last_ponctual_observation[1][1]=self.production_norm[self.counter]
        i=1
        if(self._dist_equinox==1):
            i=i+1
            self._last_ponctual_observation[i]=abs(self.counter/24-(365./2))/(365./2) #171 days between 1jan and 21 Jun
        if (self._pred==1):
            i=i+1
            self._last_ponctual_observation[i][0]=sum(self.production_norm[self.counter:self.counter+24])/24.#*self.rng.uniform(0.75,1.25)
            self._last_ponctual_observation[i][1]=sum(self.production_norm[self.counter:self.counter+48])/48.#*self.rng.uniform(0.75,1.25)
                                
        self.counter+=1
                
        return copy.copy(reward)

    def inputDimensions(self):
        return self._input_dimensions

    def nActions(self):
        return 3

    def observe(self):
        return copy.deepcopy(self._last_ponctual_observation)

    def summarizePerformance(self, test_data_set):
        print("summary perf")
        print("self.hydrogen_storage: {}".format(self.hydrogen_storage))
        observations = test_data_set.observations()
        aaa = test_data_set.actions()
        rewards = test_data_set.rewards()
        actions=[]
        for a, thea in enumerate (aaa):
            if (thea==0):
                actions.append(-self.hydrogen_max_power)
            elif (thea==1):
                actions.append(0)
            elif (thea==2):
                actions.append(self.hydrogen_max_power)

        battery_level=np.array(observations[0])*self.battery_size
        consumption=np.array(observations[1][:,0])*(self.max_consumption-self.min_consumption)+self.min_consumption
        production=np.array(observations[1][:,1])*(self.max_production-self.min_production)+self.min_production

        i=0
        plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"plot_winter_.png")

        i=180*24
        plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"plot_summer_.png")

        i=360*24
        plot_op(actions[0+i:100+i],consumption[0+i:100+i],production[0+i:100+i],rewards[0+i:100+i],battery_level[0+i:100+i],"plot_winter2_.png")

        
def main():
    rng = np.random.RandomState(123456)
    myenv=MyEnv(rng)

    print (myenv.observe())
    
if __name__ == "__main__":
    main()
