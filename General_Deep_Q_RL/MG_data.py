import numpy as np
import xlrd 
import pandas as pd

hours=24
months=12
days=30#30#*7+1
size=days*hours
#np.set_printoptions(threshold='nan')

def get_production(start_timesteps,timesteps):        
    """ Obtain PV production

    Arguments:
        start_timesteps - Production is from January 1st + start_timesteps (hours)
        timesteps - Number of timesteps (hours) to retrieve        
    """
    production=get_prod_solar_BE(start_timesteps,timesteps)
    
    maxp=np.max(production)
    production=production/maxp
    
    return production, 0, maxp, 



def get_consumption(timesteps):        
    """ Obtain Microgrid consumption

    Arguments:
        timesteps - Number of timesteps (hours) to retrieve        
    """
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
            
    consumption=consumption/3/1000 # Wh --> kWh
    
    maxc=np.max(consumption)
    consumption=consumption/maxc
    
    return consumption, 0, maxc, 

#def get_market_price():
#    market_price=read_excell(3*365,24).flatten()
#    print market_price.shape
#    
#
#    price_train=market_price[0:24*365]
#    
#    # Curtail 1% of highest prices and 1% of lowest prices
#    indices_max=price_train.argsort()[-365*24/100:][::-1]
#    indices_min=price_train.argsort()[:365*24/100][::-1]
#    price_train[indices_max]=price_train[indices_max[-1]]    
#    price_train[indices_min]=price_train[indices_min[0]]    
#    
#    print price_train[0:10]
#    min_price=price_train[indices_min[0]]
#    max_price=price_train[indices_max[-1]]
#    price_train= (price_train-price_train[indices_min[0]]) / (price_train[indices_max[-1]]-price_train[indices_min[0]]) #* 2 - 1
#    print price_train[0:10]
#    
#    price_valid=market_price[24*365+1:2*24*365]
#    price_test=market_price[2*24*365+1:3*24*365]
#    
#    print np.sum(price_train)/365/24 ## euro/MWh
#    print np.sum(price_valid)/365/24 ## euro/MWh
#    print np.sum(price_test)/365/24 ## euro/MWh
#    
#    return price_train, min_price, max_price


        
def read_excell(i,j):
    book = xlrd.open_workbook("data/spotmarket_data_2007-2013.xls")
    sh = book.sheet_by_index(1)
    
    row=np.zeros((i, j),np.float64);
    
    for rx in range(365*3,i+365*3):
        for ry in range(0,j):
        	row[rx-365*3,ry]=sh.cell_value(rx+1,ry+1)
        
    return row
    
# Spain
#def read_excell_solar(i,j):
#	book = xlrd.open_workbook("data/SolarGIS-15min-PSA-ES.xls")
#	print "The number of worksheets is", book.nsheets
#	print "Worksheet name(s):", book.sheet_names()
#	sh = book.sheet_by_index(0)
#	print sh.name, sh.nrows, sh.ncols
#	print "Cell C80 is", sh.cell_value(rowx=79, colx=2)
#	
#	row=np.zeros(i/4,np.float32);
#	
#	ry=j
#	
#	prem_janv=1 #prem janvier 2010
#	prem_mai=4*30*24*4
#	for rx in range(0,i): #FIXME2
#	    row[rx/4]+=sh.cell_value(rowx=(rx+prem_mai)%(12*30*24*4)+1,colx=(ry+10*((rx+prem_mai)/(12*30*24*4))))
#	    
#	    row[rx/4]=row[rx/4]*np.cos(   np.pi/6 * np.cos(  np.pi/2*((rx+prem_mai)%(12*30*24*4)-(6*30*24*4)) / (6*30*24*4)  )   )
#		
#	return row
#
#
#def get_prod_solar_SP(timesteps):
#    GHI=read_excell_solar(timesteps*4,2)
#    
#    production=GHI*0.065#0.075
#
#    return production
    
    
def get_prod_solar_BE(start_timesteps,timesteps):
    # Belgium
    df = pd.read_csv("data/data_Belgium.txt", sep="\t", dtype={'Production': 'double'})
    df.Date = pd.to_datetime(df.Date)
    
    date = df['Date']
    prod = df['Production']
    prod = np.transpose(np.array([prod.values]))
    
    production=np.zeros(timesteps)
    
    # data set 1 --> 8405 = 1juin  (10360=21 juin)
    # data set 2 --> 10886=1janvier 2010
    # data set 3 --> 900=1janvier 2010
    # data set 5 --> 7524=1janvier 2010
    prem_janv=7524+start_timesteps*4
    for i in range (prem_janv,prem_janv+4*timesteps):
        production[(i-prem_janv)/(4)] += prod[i]/4 /2 #(/4=per hour, /2 1m^2)
        
    return production




def main():
    timesteps=365*24#24*days*months
    prod_solar_BE=get_prod_solar_BE(3*timesteps)
    
    
if __name__ == "__main__":
    main()