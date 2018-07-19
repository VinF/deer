""" Interface with the catcher environment

"""
import numpy as np
import cv2

from deer.base_classes import Environment

import matplotlib
matplotlib.use('qt5agg')
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
plt.switch_backend('agg') # For remote servers
import copy 

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._actions = [0,1]
        self._height=10#15
        self._width=10 #preferably an odd number so that it's symmetrical
        self._width_paddle=1
        self._nx_block=2#self._width#2 #number of different x positions of the falling blocks
        self._higher_dim_obs=kwargs["higher_dim_obs"]
        self._reverse=kwargs["reverse"]

        if(self._nx_block==1):
            self._x_block=self._width//2
        else:
            rand=np.random.randint(self._nx_block) # random selection of the pos for falling block
            self._x_block=rand*((self._width-1)//(self._nx_block-1)) # traduction in a number in [0,self._width] of rand

                
    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                np.random.seed(seed=11) #Seed the generator so that the sequence of falling blocks is the same in validation
            else:
                self._mode_episode_count += 1
        elif self._mode != -1: # and thus mode == -1
            self._mode = -1
        
        self.y=self._height-1
        self.x=np.random.randint(self._width-self._width_paddle+1) #self._width//2
        if(self._nx_block==1):
            self._x_block=self._width//2
        else:
            rand=np.random.randint(self._nx_block) # random selection of the pos for falling block
            self._x_block=rand*((self._width-1)//(self._nx_block-1)) # traduction in a number in [0,self._width] of rand
                
        return [1 * [self._height * [self._width * [0]]]] #[0,0,1]+[0,1,0]
        
        
    def act(self, action):
        action = self._actions[action]
        
        if(action==0):        
            self.x = max(self.x-1,0)
        if(action==1):        
            self.x = min(self.x+1,self._width-self._width_paddle)

        self.y = self.y-1
              
        if(self.y==0 and self.x>self._x_block-1-self._width_paddle and self.x<=self._x_block+1):
            self.reward = 1
        elif(self.y==0):
            self.reward = -1
        else:
            self.reward = 0

        self._mode_score += self.reward
        return self.reward

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        #print "test_data_set.observations.shape"
        #print test_data_set.observations()[0][0:1]
        
        all_possib_inp=[]
        for x_b in range(self._nx_block):#[1]:#range(self._nx_block):
            for y_b in range(self._height):
                for x_p in range(self._width-self._width_paddle+1):
                    state=self.get_observation(y_b,x_b*((self._width-1)//(self._nx_block-1)),x_p)
                    all_possib_inp.append(state)

        all_possib_inp=np.expand_dims(all_possib_inp,axis=1)
        print "all_possib_inp"
        print all_possib_inp[0]
        print all_possib_inp[self._height*(self._width-self._width_paddle+1)-1]
        print all_possib_inp[self._height*(self._width-self._width_paddle+1)]
        print all_possib_inp[2*self._height*(self._width-self._width_paddle+1)-1]
        #print all_possib_inp[2*self._height*(self._width-self._width_paddle+1)]
        #print all_possib_inp[3*self._height*(self._width-self._width_paddle+1)-1]
        print "all_possib_inp.shape"
        print all_possib_inp.shape
        #print all_possib_inp[self._height*self._width]
        #print "all_possib_inp[2*self._height*self._width]"
        #print all_possib_inp[2*self._height*self._width]
        #print all_possib_inp[2*self._height*self._width-1]
        all_possib_abs_states=learning_algo.encoder.predict(all_possib_inp)
        print "learning_algo.encoder.predict(all_possib_inp)"
        print all_possib_abs_states
        
        #print "print test_data_set.observations()"
        #print test_data_set.observations()
        n=self._height-1
        historics=[]
        for i,observ in enumerate(test_data_set.observations()[0][0:n]):
            historics.append(np.expand_dims(observ,axis=0))
        historics=np.array(historics)
        print "historics"
        print historics
        abs_states=learning_algo.encoder.predict(historics)
        print "abs_states"
        print abs_states
        actions=test_data_set.actions()[0:n]
        print "actions"
        print actions

        print actions
        print "test_data_set.rewards()[0:n]"
        print test_data_set.rewards()[0:n]
        print "test_data_set.terminals()[0:n]"
        print test_data_set.terminals()[0:n]
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))
                
        
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cm.jet)
        
        x = np.array(abs_states)[:,0]
        y = np.array(abs_states)[:,1]
        z = np.array(abs_states)[:,2]
        
        #Colors
        #onehot_actions = np.zeros((n, 4))
        #onehot_actions[np.arange(n), actions] = 1
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_xlabel(r'$X_1$')
        ax.set_ylabel(r'$X_2$')
        ax.set_zlabel(r'$X_3$')

        for j in range(3):
            # Plot the trajectory
            for i in xrange(30):#(n-1):
                ax.plot(x[j*24+i:j*24+i+2], y[j*24+i:j*24+i+2], z[j*24+i:j*24+i+2], color=plt.cm.cool(255*i/n), alpha=0.5)

        # Plot the estimated transitions
        for i in range(n-1):
            predicted1=learning_algo.transition.predict([abs_states[i:i+1],np.array([[1,0]])])
            predicted2=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,1]])])
            ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.75", alpha=0.75)
            ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.25", alpha=0.75)

#        for xx in np.arange(self._width)-self._width//2:
#            for yy in np.arange(self._width)-self._width//2:
#                for zz in np.arange(self._width)-self._width//2:
#                    predicted1=learning_algo.transition.predict([np.array([[xx,yy,zz]]),np.array([[1,0]])])
#                    predicted2=learning_algo.transition.predict([np.array([[xx,yy,zz]]),np.array([[0,1]])])
#                    ax.plot(np.concatenate([np.array([xx]),predicted1[0,:1]]), np.concatenate([np.array([yy]),predicted1[0,1:2]]), np.concatenate([np.array([zz]),predicted1[0,2:]]), color="1", alpha=0.5)
#                    ax.plot(np.concatenate([np.array([xx]),predicted2[0,:1]]), np.concatenate([np.array([yy]),predicted2[0,1:2]]), np.concatenate([np.array([zz]),predicted2[0,2:]]), color="0.5", alpha=0.5)
        

        # Plot the colorbar for the trajectory
        fig.subplots_adjust(right=0.7)
        ax1 = fig.add_axes([0.725, 0.15, 0.025, 0.7])
        # Set the colormap and norm to correspond to the data for which the colorbar will be used.
        cmap = matplotlib.cm.cool
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)

        # ColorbarBase derives from ScalarMappable and puts a colorbar in a specified axes, so it has 
        # everything needed for a standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks and labels.
        cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm,
                                orientation='vertical')
        cb1.set_label('Beginning to end of trajectory')


        # Plot the dots at each time step depending on the action taken
        length_block=self._height*(self._width-self._width_paddle+1)
        for i in range(self._nx_block):
            line3 = ax.scatter(all_possib_abs_states[i*length_block:(i+1)*length_block,0], all_possib_abs_states[i*length_block:(i+1)*length_block,1] ,all_possib_abs_states[i*length_block:(i+1)*length_block,2], s=10, marker='x', depthshade=True, edgecolors='k', alpha=0.3)
        line2 = ax.scatter(x, y ,z , c=np.tile(np.expand_dims(1-actions/2.,axis=1),(1,3))-0.25, s=50, marker='o', edgecolors='k', alpha=0.75, depthshade=True)
        axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
        zrange=axes_lims[2][1]-axes_lims[2][0]
        
        # Plot the legend for the dots
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
        box1 = TextArea(" State representation (action 0, action 1): ", textprops=dict(color="k"))
        
        box2 = DrawingArea(60, 20, 0, 0)
        el1 = Circle((10, 10), 5, fc="0.75", edgecolor="k", alpha=0.75)
        el2 = Circle((30, 10), 5, fc="0.25", edgecolor="k", alpha=0.75) 
        #el3 = Circle((50, 10), 5, fc="0", edgecolor="k") 
        box2.add_artist(el1)
        box2.add_artist(el2)
        #box2.add_artist(el3)


        box = HPacker(children=[box1, box2],
                      align="center",
                      pad=0, sep=5)
        
        anchored_box = AnchoredOffsetbox(loc=3,
                                         child=box, pad=0.,
                                         frameon=True,
                                         bbox_to_anchor=(0., 1.07),
                                         bbox_transform=ax.transAxes,
                                         borderpad=0.,
                                         )
        ax.add_artist(anchored_box)


        # Plot the legend for transition estimates
        box1b = TextArea(" Estimated transitions (action 0, action 1): ", textprops=dict(color="k"))
        box2b = DrawingArea(60, 20, 0, 0)
        el1b = Rectangle((5, 10), 15,2, fc="0.75", alpha=0.75)
        el2b = Rectangle((25, 10), 15,2, fc="0.25", alpha=0.75) 
        box2b.add_artist(el1b)
        box2b.add_artist(el2b)

        boxb = HPacker(children=[box1b, box2b],
                      align="center",
                      pad=0, sep=5)
        
        anchored_box = AnchoredOffsetbox(loc=3,
                                         child=boxb, pad=0.,
                                         frameon=True,
                                         bbox_to_anchor=(0., 0.98),
                                         bbox_transform=ax.transAxes,
                                         borderpad=0.,
                                         )        
        ax.add_artist(anchored_box)

        

        ax.w_xaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
        ax.w_yaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
        ax.w_zaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
        #plt.savefig('fig_base'+str(learning_algo.update_counter)+'.pdf')


        # Plot the Q_vals
        c = learning_algo.Q.predict(np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1))
        #print "actions,C"
        #print actions
        #print c
        #c=np.max(c,axis=1)
        m1=ax.scatter(x, y, z+zrange/20, c=c[:,0], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
        m2=ax.scatter(x, y, z+3*zrange/40, c=c[:,1], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
        
        #plt.colorbar(m3)
        ax2 = fig.add_axes([0.85, 0.15, 0.025, 0.7])
        cmap = matplotlib.cm.RdYlGn
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
        cb1.set_label('Estimated expected return')

        plt.show()
        for ii in xrange(-15,345,30):
            ax.view_init(elev=20., azim=ii)
            plt.savefig('fig_w_V_div5_'+str(learning_algo.update_counter)+'_'+str(ii)+'.pdf')


        # fig_visuV
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([i for i in range(5) for jk in range(25)])/4.*(axes_lims[0][1]-axes_lims[0][0])+axes_lims[0][0]
        y = np.array([j for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[1][1]-axes_lims[1][0])+axes_lims[1][0]
        z = np.array([k for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[2][1]-axes_lims[2][0])+axes_lims[2][0]

        c = learning_algo.Q.predict(np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1))
        c=np.max(c,axis=1)
        #print "c"
        #print c
        
        m=ax.scatter(x, y, z, c=c, vmin=-1., vmax=1., cmap=plt.hot())
        #plt.colorbar(m)
        fig.subplots_adjust(right=0.8)
        ax2 = fig.add_axes([0.875, 0.15, 0.025, 0.7])
        cmap = matplotlib.cm.hot
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
        cb1.set_label('Estimated expected return')

        #plt.show()
        #plt.savefig('fig_visuV'+str(learning_algo.update_counter)+'.pdf')


        # fig_visuR
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([i for i in range(5) for jk in range(25)])/4.*(axes_lims[0][1]-axes_lims[0][0])+axes_lims[0][0]
        y = np.array([j for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[1][1]-axes_lims[1][0])+axes_lims[1][0]
        z = np.array([k for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[2][1]-axes_lims[2][0])+axes_lims[2][0]

        coords=np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1)
        repeat_nactions_coord=np.repeat(coords,self.nActions(),axis=0)
        identity_matrix = np.diag(np.ones(self.nActions()))
        tile_identity_matrix=np.tile(identity_matrix,(5*5*5,1))

        c = learning_algo.R.predict([repeat_nactions_coord,tile_identity_matrix])
        c=np.max(np.reshape(c,(125,self.nActions())),axis=1)
        #print "c"
        #print c
        #mini=np.min(c)
        #maxi=np.max(c)
        
        m=ax.scatter(x, y, z, c=c, vmin=-1., vmax=1., cmap=plt.hot())
        #plt.colorbar(m)
        fig.subplots_adjust(right=0.8)
        ax2 = fig.add_axes([0.875, 0.15, 0.025, 0.7])
        cmap = matplotlib.cm.hot
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)

        # ColorbarBase derives from ScalarMappable and puts a colorbar
        # in a specified axes, so it has everything needed for a
        # standalone colorbar.  There are many more kwargs, but the
        # following gives a basic continuous colorbar with ticks
        # and labels.
        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
        cb1.set_label('Estimated expected return')

        #plt.show()
        #plt.savefig('fig_visuR'+str(learning_algo.update_counter)+'.pdf')

        matplotlib.pyplot.close("all") # avoids memory leaks

    def inputDimensions(self):
        if(self._higher_dim_obs==True):
            return [(1,(self._height+2)*3,(self._width+2)*3)]
        else:
            return [(1,self._height,self._width)]
        

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        obs=self.get_observation(self.y,self._x_block,self.x)
        return [obs]

    def get_observation(self,y,x_block,x):
        obs=np.zeros((self._height,self._width))
        obs[y,x_block]=0.5
        obs[0,x-self._width_paddle+1:x+1]=1
        
        if(self._higher_dim_obs==True):
            y_t=(1+y)*3
            x_block_t=(1+x_block)*3
            x_t=(1+x)*3
            obs=np.zeros(( (self._height+2)*3 , (self._width+2)*3 ))
            ball=np.array([[0,0,0.6,0.8,0.6,0,0],
                            [0.,0.6,0.9,1,0.9,0.6,0],
                            [0.,0.85,1,1,1,0.85,0.],
                            [0,0.6,0.9,1,0.9,0.6,0],
                            [0,0,0.6,0.85,0.6,0,0]])
            paddle=np.array([[0.5,0.95,1,1,1,0.95,0.5],
                            [0.9,1,1,1,1,1,0.9],
                            [0.,0.,0,0,0,0.,0.]])
            
            obs[y_t-2:y_t+3,x_block_t-3:x_block_t+4]=ball
            obs[3:6,x_t-3:x_t+4]=paddle
        
        if(self._reverse==True):
            obs=-obs
            #plt.imshow(np.flip(obs,axis=0), cmap='gray_r')
            #plt.show()

        return obs

    def inTerminalState(self):
        if (self.y==0):
            return True
        else:
            return False



if __name__ == "__main__":
    pass
