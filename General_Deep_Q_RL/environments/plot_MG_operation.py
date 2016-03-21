from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import numpy as np


def plot_op(actions, consumption,production,rewards,battery_level, plot_name):
    ####
    # PLOT
    ####
    
    print ( "In this plot, total score"+str(np.sum(rewards)) )
    print ( "H:"+str( np.sum(actions)*0.1 ) )
    print ( "-:"+str( np.sum(rewards)-np.sum(actions)*0.1 ) )
    
    print ("battery_level")
    print (battery_level)
    print ("actions")
    print (actions)
    print ("consumption")
    print (consumption)
    print ("production")
    print (production)
    print ("rewards")
    print (rewards)
    
    steps=np.arange(100)
    print (steps)
    print ("battery_level")
    print (battery_level)
    print (consumption)
    print (production)
    print ("demand:")
    print (consumption-production)
    
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
    host.set_ylim(0, 20.9)
    
    host.set_xlabel("Time (h)")
    host.set_ylabel("Battery level (kWh)")
    par1.set_ylabel("Consumption (kW)")
    par2.set_ylabel("Production (kW)")
    par3.set_ylabel("H Actions (kW)")
    
    p1, = host.plot(steps, battery_level, marker='o', lw=1, c = 'b', alpha=0.8, ls='-', label = 'Battery level')
    p2, = par1.plot(steps_long-0.9, np.repeat(consumption,10), lw=3, c = 'r', alpha=0.5, ls='-', label = 'Consumption')
    p3, = par2.plot(steps_long-0.9, np.repeat(production,10), lw=3, c = 'g', alpha=0.5, ls='-', label = 'Production')
    p4, = par3.plot(steps_long, np.repeat(actions,10), lw=3, c = 'c', alpha=0.5, ls='-', label = 'H Actions')
    
    par1.set_ylim(0, 10.09)
    par2.set_ylim(0, 10.09)
    par3.set_ylim(-1.5, 1.5)
    
    #host.legend(loc=2)#loc=9)
    
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    par2.axis["right"].label.set_color(p3.get_color())
    par3.axis["right"].label.set_color(p4.get_color())
    
    plt.savefig(plot_name)
    
    #plt.draw()
    #plt.show()
    #plt.close('all')
