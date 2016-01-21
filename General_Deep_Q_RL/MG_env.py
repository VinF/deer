import numpy as np
import xlrd 
np.set_printoptions(threshold=np.nan)

class Env(object):
    def __init__(self, rng):
        """ Initialize environment

        Arguments:
            rng - the numpy random number generator
            
        """
        self.market_price, self.min_market_price, self.max_market_price=get_market_price()
        print "self.market_price: "+str(self.market_price[0:100])
        print self.min_market_price, self.max_market_price, self.market_price.shape
        
        self.consumption, self.min_consumption, self.max_consumption=get_consumption(365*24)
        print "self.consumption: " + str(self.consumption[0:100])
        print self.min_consumption, self.max_consumption, self.consumption.shape

        self.production, self.min_production, self.max_production=get_production(365*24)
        print "self.production: " + str(self.production[0:100])
        print self.min_production, self.max_production, self.production.shape


        self.rng = rng

#        # test 1
#        self.observation=[0] # List of k elements of an observation (eg h*w, w or float)
#        self.num_elements_in_batch=[1]
#        
#        # test 2
#        self.observation=[(self.rng.uniform(0, 1., size=(4))/256).tolist(),0] # List of k elements of an observation (eg h*w, w or float)
#        self.num_elements_in_batch=[3,1]
        

#        # test 3
#        self.observation=[0.5,0.5,0.5]
#        self.num_elements_in_batch=[1,1,4] # [battery storage, network_cost_short, network_cost_long]

        # test 4
#        self.battery_max_power=2.
#        self.battery_size=10.
#        self.battery_eta=0.9
#        
#        self.observation=[0.,0.,0.,0,0]
#        self.num_elements_in_batch=[1,1,4,1,4] # [battery storage, network_cost_short, network_cost_long, consumption_short, consumption_long]
        
        
        
#        # test 5
#        self.battery_max_power=2.
#        self.battery_size=10.
#        self.battery_eta=0.9
#        
#        self.observation=[0.,0.,0.,0,0,0,0]
#        self.num_elements_in_batch=[1,1,4,1,4,1,4] # [battery storage, network_cost_short, network_cost_long, consumption_short, consumption_long]
#        self.reward_mean_running=-75.0
#        self.reward_std_running=80.#100
        
        # test 6
        self.battery_max_power=2.
        self.battery_size=10.
        self.battery_eta=0.9
        
        self.observation=[0.,0.,0.,0,0,0,0]
        self.num_elements_in_batch=[1,1,4,1,4,1,4] # [battery storage, network_cost_short, network_cost_long, consumption_short, consumption_long, production_short, production_long]
        self.reward_mean_running=-75.0
        self.reward_std_running=80.#100
        
        
    def init(self, testing):
        """
        Returns:
           current observation (list of k elements)
        """
        ### Test 3
        #self.observation=[0.5,0.5,0.5]
                
        ### Test 4
        #self.observation=[0.,0.,0.,0.,0.]
        
        ### Test 6
        self.observation=[0.,0.,0.,0.,0.,0.,0.]
        
        self.counter = 0 ## reinitialize counter after each epoch
        

        return self.observation
        
        
    def act(self, action, testing):
        """
        Perform one time step on the environment
        """
        reward = 0#self.ale.act(action)  #FIXME
        terminal=0
        #rng = np.random.RandomState(123456)        


#        ### Test 1
#        if (action==0):
#            self.observation[0]=max(self.observation[0]-1,0)
#        if (action==1):
#            self.observation[0]=self.observation[0]+1
#            
#        if (self.observation[0]==5):
#            reward=1
#
#        if (self.observation[0]>10):
#            terminal=1
#            self.observation[0]=0
        
#        ### Test 2       
#        if (action==0):
#            self.observation[1]=max(self.observation[1]-1,0)
#        if (action==1):
#            self.observation[1]=self.observation[1]+1
#            
#        if (self.observation[1]==5):
#            reward=1
#
#        if (self.observation[1]>10):
#            terminal=1
#            self.observation[1]=0
        
#        ### Test 3 
#        ## NB use market price *2 -1 to ease the process (line 224!)     
#        if (action==0):
#            reward=self.market_price[self.counter-1]*min(self.observation[0],0.1)*10
#            self.observation[0]=max(self.observation[0]-0.1,0)
#        if (action==1):
#            reward=-self.market_price[self.counter-1]*min(1-self.observation[0],0.1)*10
#            self.observation[0]=min(self.observation[0]+0.1,1)
#
#        self.observation[1]=self.market_price[self.counter]
#        self.observation[2]=self.market_price[self.counter]#self.observation[1].extend(self.market_price[self.counter])[0:12]

        ### Test 4
        ### use market price *2 -1 to ease the process (line 224!) 
#        # Case with one battery, a variable grid price, an internal consumption
#        # self.observation[0] : battery state (normalised [0,1])
#        # self.observation[1] : market_price_short (normalised [-1,1])
#        # self.observation[2] : market_price_long (normalised [-1,1])
#        # self.observation[3] : consumption_short (normalised [-1,1])
#        # self.observation[3] : consumption_long (normalised [-1,1])
#        ###
#        
#        true_market_price=self.market_price[self.counter-1]*(self.max_market_price-self.min_market_price)+self.min_market_price
#        true_cons=self.consumption[self.counter]*(self.max_consumption-self.min_consumption)+self.min_consumption
#               
#        #print "action:"+str(action)
#        #print "state b"+str(self.observation[0])
#        #print "true_cons"+str(true_cons)
#        if (action==0):
#            ## Energy is taken out of the battery
#            true_energy_avail_from_battery=min(self.observation[0]*self.battery_size,self.battery_max_power) #(>0)
#            true_energy_avail_from_MG=true_energy_avail_from_battery*self.battery_eta-true_cons
#        if (action==1):
#            ## Energy is taken into the battery
#            true_energy_avail_from_battery=-min( (1-self.observation[0])*self.battery_size,self.battery_max_power) #(<0)
#            true_energy_avail_from_MG=true_energy_avail_from_battery/self.battery_eta-true_cons
#        
#        #print "true_energy_avail_from_battery:"+str(true_energy_avail_from_battery)
#        #print "true_energy_avail_from_MG:"+str(true_energy_avail_from_MG)
#        energy_avail_from_battery=true_energy_avail_from_battery/self.battery_max_power # (normalised [-1,1]) NB: true_cons/self.battery_eta<self.battery_max_power
#        energy_avail_from_MG=true_energy_avail_from_MG/self.battery_max_power #(normalised)
#
#
#        reward=self.market_price[self.counter-1]*energy_avail_from_MG
#        #print "reward=self.market_price[self.counter-1] * true_energy_avail_from_MG"
#        #print str(reward)+"="+str(self.market_price[self.counter-1])+"*"+str(energy_avail_from_MG)
#            
##        print "state b before:"+str(self.observation[0])
#        self.observation[0]=self.observation[0]-true_energy_avail_from_battery/self.battery_size
##        print "state b after:"+str(self.observation[0])
#        
#
#        self.observation[1]=self.market_price[self.counter]#self.observation[1].extend(self.market_price[self.counter])[0:12]
#        self.observation[2]=self.market_price[self.counter]
#        self.observation[3]=self.consumption[self.counter]
#        self.observation[4]=self.consumption[self.counter]


        ### Test 5
        ### same as test4 but scaling reward with running average
        # Case with one battery, a variable grid price, an internal consumption
        # self.observation[0] : battery state (normalised [0,1])
        # self.observation[1] : market_price_short (normalised [-1,1])
        # self.observation[2] : market_price_long (normalised [-1,1])
        # self.observation[3] : consumption_short (normalised [-1,1])
        # self.observation[3] : consumption_long (normalised [-1,1])
        ###
#        
#        true_market_price=self.market_price[self.counter-1]*(self.max_market_price-self.min_market_price)+self.min_market_price
#        true_cons=self.consumption[self.counter]*(self.max_consumption-self.min_consumption)+self.min_consumption
#               
#        #print "action:"+str(action)
#        #print "state b"+str(self.observation[0])
#        #print "true_cons"+str(true_cons)
#        if (action==0):
#            ## Energy is taken out of the battery
#            true_energy_avail_from_battery=min(self.observation[0]*self.battery_size,self.battery_max_power) #(>0)
#            true_energy_avail_from_MG=true_energy_avail_from_battery*self.battery_eta-true_cons
#        if (action==1):
#            ## Energy is taken into the battery
#            true_energy_avail_from_battery=-min( (1-self.observation[0])*self.battery_size,self.battery_max_power) #(<0)
#            true_energy_avail_from_MG=true_energy_avail_from_battery/self.battery_eta-true_cons
#        
#        reward=true_market_price*true_energy_avail_from_MG
#        print "reward=true_market_price*true_energy_avail_from_MG"
#        print str(reward)+"="+str(true_market_price)+"*"+str(true_energy_avail_from_MG)
#            
##        print "state b before:"+str(self.observation[0])
#        self.observation[0]=self.observation[0]-true_energy_avail_from_battery/self.battery_size
##        print "state b after:"+str(self.observation[0])
#        
#
#        self.observation[1]=self.market_price[self.counter]#self.observation[1].extend(self.market_price[self.counter])[0:12]
#        self.observation[2]=self.market_price[self.counter]
#        self.observation[3]=self.consumption[self.counter]
#        self.observation[4]=self.consumption[self.counter]




        ### Test 6
        ### same as test5 but with 30m^2 of PV
        # Case with one battery, a variable grid price, an internal consumption
        # self.observation[0] : battery state (normalised [0,1])
        # self.observation[1] : market_price_short (normalised [-1,1])
        # self.observation[2] : market_price_long (normalised [-1,1])
        # self.observation[3] : consumption_short (normalised [-1,1])
        # self.observation[3] : consumption_long (normalised [-1,1])
        ###
        
        true_market_price=self.market_price[self.counter-1]*(self.max_market_price-self.min_market_price)+self.min_market_price
        true_cons=self.consumption[self.counter]*(self.max_consumption-self.min_consumption)+self.min_consumption
        true_prod=self.production[self.counter]*(self.max_production-self.min_production)+self.min_production
        true_prod=true_prod*30/1000 #(30m^2 et en kWh)
        
        true_cons=true_cons-true_prod
        
        #print "action:"+str(action)
        #print "state b"+str(self.observation[0])
        #print "true_cons"+str(true_cons)
        if (action==0):
            ## Energy is taken out of the battery
            true_energy_avail_from_battery=min(self.observation[0]*self.battery_size,self.battery_max_power) #(>0)
            true_energy_avail_from_MG=true_energy_avail_from_battery*self.battery_eta-true_cons
        if (action==1):
            ## Energy is taken into the battery
            true_energy_avail_from_battery=-min( (1-self.observation[0])*self.battery_size,self.battery_max_power) #(<0)
            true_energy_avail_from_MG=true_energy_avail_from_battery/self.battery_eta-true_cons
        
        reward=true_market_price*true_energy_avail_from_MG
        print "reward=true_market_price*true_energy_avail_from_MG"
        print str(reward)+"="+str(true_market_price)+"*"+str(true_energy_avail_from_MG)
            
#        print "state b before:"+str(self.observation[0])
        self.observation[0]=self.observation[0]-true_energy_avail_from_battery/self.battery_size
#        print "state b after:"+str(self.observation[0])
        

        self.observation[1]=self.market_price[self.counter]#self.observation[1].extend(self.market_price[self.counter])[0:12]
        self.observation[2]=self.market_price[self.counter]
        self.observation[3]=self.consumption[self.counter]
        self.observation[4]=self.consumption[self.counter]
        self.observation[5]=self.production[self.counter]
        self.observation[6]=self.production[self.counter]

           
        #if (self.counter>510):
        #    terminal=1
        #    self.observation[1]=0
        #    self.counter=0           
            
            
        self.counter+=1
        
        #print "reward, self.observation, terminal"
        #print reward, self.observation, terminal
        
        #self.reward_mean_running=self.reward_mean_running*0.999+0.001*reward
        #self.reward_std_running=self.reward_std_running*0.999+0.001*((self.reward_mean_running-reward)**2)**0.5
        
        reward_scaled=(reward-self.reward_mean_running)/self.reward_std_running
        #print "reward=self.market_price[self.counter-1] * true_energy_avail_from_MG"
        print "reward_scaled=(reward-self.reward_mean_running)/self.reward_std_running"
        print str(reward_scaled)+"=("+str(reward)+"-"+str(self.reward_mean_running)+")/"+str(self.reward_std_running)

        
        if (testing):
            return reward, self.observation, terminal
        else:
            
            return reward_scaled, self.observation, terminal



# Spain
def read_excell_solar(i,j):
	book = xlrd.open_workbook("data/SolarGIS-15min-PSA-ES.xls")#("spotmarket_data_2011-2013.xls")  #("spotmarket_data_2009-2013.xls") #("spotmarket_data_2011-2013.xls")
	print "The number of worksheets is", book.nsheets
	print "Worksheet name(s):", book.sheet_names()
	sh = book.sheet_by_index(0)
	print sh.name, sh.nrows, sh.ncols
	print "Cell C80 is", sh.cell_value(rowx=79, colx=2)
	
	row=np.zeros(i/4,np.float32);
	
	ry=j
	
	prem_janv=1 #prem janvier 2010
	prem_mai=4*30*24*4
	for rx in range(0,i): #FIXME2
	    row[rx/4]+=sh.cell_value(rowx=(rx+prem_mai)%(12*30*24*4)+1,colx=(ry+10*((rx+prem_mai)/(12*30*24*4))))
	    
	    row[rx/4]=row[rx/4]*np.cos(   np.pi/6 * np.cos(  np.pi/2*((rx+prem_mai)%(12*30*24*4)-(6*30*24*4)) / (6*30*24*4)  )   )
		
	return row

        
def get_production(timesteps):        
    GHI=read_excell_solar(timesteps*4,2)
    
    production=GHI*0.065#0.075
    
    maxp=np.max(production)
    production=production/maxp
    
    return production, 0, maxp, 



def get_consumption(timesteps):        
    ###
    ## Consumption profile epuration ##    
    consumption=np.zeros(timesteps)
    mu1=14 # consumption centered on mu1
    for i in range (0,timesteps):
        if (i%(24)>=9 and i%(24)<9+6):      #Tube TLed-60 cm-18 Watt-9 W reel
            consumption[i]+=20*20
        if (i%(24)>=9 and i%(24)<9+8):      #Ampoules LED 7 Watts
            consumption[i]+=50*7
        if (i%(24)>=9 and i%(24)<9+4):      #TV's
            consumption[i]+=6*100
        if (i%(24)>=9 and i%(24)<9+8):      #ordis
            consumption[i]+=2*100+2*150
    
        consumption[i]+=2*250+1*370         #pompes doseuses
        if (i%(24)>=9 and i%(24)<9+4):      #agitateur
            consumption[i]+=2*250+1*350
        if (i%(24)>=7 and i%(24)<9+12):     #spot led
            consumption[i]+=10*50
        if (i%(24)>=8 and i%(24)<9+10):     #surpresseur
            consumption[i]+=1*1200
        consumption[i]+=1*150               #autre
        if (i%(24)>=9 and i%(24)<9+7):     #frigo
            consumption[i]+=3*150
            
    consumption=consumption/1000 # Wh --> kWh
    
    maxc=np.max(consumption)
    consumption=consumption/maxc
    
    return consumption, 0, maxc, 

def get_market_price():
    market_price=read_excell(3*365,24).flatten()
    print market_price.shape
    

    price_train=market_price[0:24*365]
    
    # Curtail 1% of highest prices and 1% of lowest prices
    indices_max=price_train.argsort()[-365*24/100:][::-1]
    indices_min=price_train.argsort()[:365*24/100][::-1]
    price_train[indices_max]=price_train[indices_max[-1]]    
    price_train[indices_min]=price_train[indices_min[0]]    
    
    print price_train[0:10]
    min_price=price_train[indices_min[0]]
    max_price=price_train[indices_max[-1]]
    price_train= (price_train-price_train[indices_min[0]]) / (price_train[indices_max[-1]]-price_train[indices_min[0]]) #* 2 - 1
    print price_train[0:10]
    
    price_valid=market_price[24*365+1:2*24*365]
    price_test=market_price[2*24*365+1:3*24*365]
    
    print np.sum(price_train)/365/24 ## euro/MWh
    print np.sum(price_valid)/365/24 ## euro/MWh
    print np.sum(price_test)/365/24 ## euro/MWh
    
    return price_train, min_price, max_price


        
def read_excell(i,j):
    book = xlrd.open_workbook("data/spotmarket_data_2007-2013.xls")
    sh = book.sheet_by_index(1)
    
    row=np.zeros((i, j),np.float64);
    
    for rx in range(365*3,i+365*3):
        for ry in range(0,j):
        	row[rx-365*3,ry]=sh.cell_value(rx+1,ry+1)
        
    return row



def main():
    rng = np.random.RandomState(123456)
    myenv=Env(rng)

    print "market price"

    aa, minaa,maxaa=get_market_price()
    print aa[0:100], minaa,maxaa
    


    print "consumption"
    aa, minaa,maxaa=get_consumption(100)
    print aa[0:100], minaa,maxaa
    
    print "production"
    aa, minaa,maxaa=get_production(100)
    print aa[0:100], minaa,maxaa
    
    
    print myenv.act(1, False)
    print myenv.act(1, False)
    print myenv.act(0, False)
    print myenv.act(0, False)
    
if __name__ == "__main__":
    main()
