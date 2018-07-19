""" Simple maze environment

Authors: Vincent Francois-Lavet

"""
import numpy as np
import cv2

from deer.base_classes import Environment

import matplotlib
#matplotlib.use('agg')
matplotlib.use('qt5agg')
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt
import copy 

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._actions = [0,1,2,3]
        self._size_maze=8
        self._higher_dim_obs=kwargs["higher_dim_obs"]
        self.create_map()
        self.intern_dim=2

    def create_map(self):
        self._map=np.zeros((self._size_maze,self._size_maze))
        self._map[-1,:]=1
        self._map[0,:]=1
        self._map[:,0]=1
        self._map[:,-1]=1
        self._map[:,self._size_maze//2]=1
        self._map[self._size_maze//2,self._size_maze//2]=0
        #self._map[:,self._size_maze//3]=1
        #self._map[-2,self._size_maze//3]=0
        #self._map[:,2*self._size_maze//3]=1
        #self._map[2,2*self._size_maze//3]=0
        self._pos_agent=[2,2]
        self._pos_goal=[self._size_maze-2,self._size_maze-2]
        #self._map[3,6]=0.66

                
    def reset(self, mode):
        self.create_map()

        self._map[self._size_maze//2,self._size_maze//2]=0
        #if mode == -1:
        #    i=np.random.randint(2)
        #    if(i==0):
        #        self._map[self._size_maze//2-1,self._size_maze//2]=0
        #    if(i==1):
        #        self._map[self._size_maze//2+1,self._size_maze//2]=0
        #else:
        #    self._map[self._size_maze//2+1,self._size_maze//2]=0
        
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:
            self._mode = -1
        
        #if self._mode == -1:
        self._pos_agent=[self._size_maze//2,self._size_maze//2]
        #else:
        #    self._pos_agent=[1,1]
            
        print "reset mode"
        print mode
        print "self._map"
        print self._map
                
        return [1 * [self._size_maze * [self._size_maze * [0]]]]
        
        
    def act(self, action):
        action = self._actions[action]
        self._cur_action=action
        if(action==0):
            if(self._map[self._pos_agent[0]-1,self._pos_agent[1]]==0):
                self._pos_agent[0]=self._pos_agent[0]-1
        elif(action==1):        
            if(self._map[self._pos_agent[0]+1,self._pos_agent[1]]==0):
                self._pos_agent[0]=self._pos_agent[0]+1
        elif(action==2):        
            if(self._map[self._pos_agent[0],self._pos_agent[1]-1]==0):
                self._pos_agent[1]=self._pos_agent[1]-1
        elif(action==3):        
            if(self._map[self._pos_agent[0],self._pos_agent[1]+1]==0):
                self._pos_agent[1]=self._pos_agent[1]+1
        
        self.reward = 0
        #if (self._pos_agent==self._pos_goal):
        #    self.reward = 1
        #if (self._pos_agent[1]>=self._size_maze-2 and action==3):
        #    self.reward = 1 # used to delete those transitions

        self._mode_score += self.reward
        return self.reward

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        #print "test_data_set.observations.shape"
        #print test_data_set.observations()[0][0:1]
        
        for i in range(1):
            all_possib_inp=[]
            self.create_map()
            for y_a in range(self._size_maze):
                for x_a in range(self._size_maze):                
                    state=copy.deepcopy(self._map)
                    state[self._size_maze//2,self._size_maze//2]=0
                    if(state[x_a,y_a]==0):
                        if(self._higher_dim_obs==True):
                            all_possib_inp.append(self.get_higher_dim_obs([[x_a,y_a]],[self._pos_goal]))
                        else:
                            state[x_a,y_a]=0.5
                            all_possib_inp.append(state)

            
            all_possib_inp=np.expand_dims(np.array(all_possib_inp,dtype='float'),axis=1)
            #print "all_possib_inp[0:2]"
            #print all_possib_inp[0:2]
            print "all_possib_inp.shape"
            print all_possib_inp.shape
            print all_possib_inp.dtype
            print all_possib_inp[0,0,:]
            print "learning_algo.encoder.predict(all_possib_inp[0:1,0:1,:])"
            print learning_algo.encoder.predict(all_possib_inp[0:1,0:1,:])
            all_possib_abs_states=learning_algo.encoder.predict(all_possib_inp)
            if(all_possib_abs_states.ndim==4):
                all_possib_abs_states=np.transpose(all_possib_abs_states, (0, 3, 1, 2))    # data_format='channels_last' --> 'channels_first'
            print "learning_algo.encoder.predict(all_possib_inp)[0:2]"
            print all_possib_abs_states[0:2]
            
            #print "print test_data_set.observations()[0:2]"
            #print test_data_set.observations()[0][0:2]
            n=500
            historics=[]
            for i,observ in enumerate(test_data_set.observations()[0][0:n]):
                historics.append(np.expand_dims(observ,axis=0))
            historics=np.array(historics)
            #print "historics[0:2]"
            #print historics[0:2]
            print "historics.shape"
            print historics.shape
            print historics.dtype
            print historics[0,0,:]
            print "learning_algo.encoder.predict(historics[0:1,0:1,:])"
            print learning_algo.encoder.predict(historics[0:1,0:1,:])
            print learning_algo.encoder.predict(all_possib_inp[0:1,0:1,:])
            print "all_possib_inp[0:1,0:1,:]==historics[0:1,0:1,:]"
            print all_possib_inp[0:1,0:1,:]==historics[0:1,0:1,:]
            abs_states=learning_algo.encoder.predict(historics)
            if(abs_states.ndim==4):
                abs_states=np.transpose(abs_states, (0, 3, 1, 2))    # data_format='channels_last' --> 'channels_first'
            print "abs_states[0:2]"
            print abs_states[0:2]
            print abs_states.shape
            actions=test_data_set.actions()[0:n]
            print "actions[0:10]"
            print actions[0:10]
            
            print "test_data_set.rewards()[0:10]"
            print test_data_set.rewards()[0:10]
            print "test_data_set.terminals()[0:10]"
            print test_data_set.terminals()[0:10]
            if self.inTerminalState() == False:
                self._mode_episode_count += 1
            print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / (self._mode_episode_count+0.0001), self._mode_episode_count))
                    
            
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.cm as cm
            m = cm.ScalarMappable(cmap=cm.jet)
            
            x = np.array(abs_states)[:,0]
            y = np.array(abs_states)[:,1]
            if(self.intern_dim>2):
                z = np.array(abs_states)[:,2]
            
            #Colors
            #onehot_actions = np.zeros((n, 4))
            #onehot_actions[np.arange(n), actions] = 1
            
            fig = plt.figure()
            if(self.intern_dim==2):
                ax = fig.add_subplot(111)
                ax.set_xlabel(r'$X_1$')
                ax.set_ylabel(r'$X_2$')
            else:
                ax = fig.add_subplot(111,projection='3d')
                ax.set_xlabel(r'$X_1$')
                ax.set_ylabel(r'$X_2$')
                ax.set_zlabel(r'$X_3$')
            
            #for j in range(3):
            #    # Plot the trajectory
            #    for i in xrange(n-1):
            #        #ax.plot(x[j*24+i:j*24+i+2], y[j*24+i:j*24+i+2], z[j*24+i:j*24+i+2], color=plt.cm.cool(255*i/n), alpha=0.5)
            #        ax.plot(x[j*24+i:j*24+i+2], y[j*24+i:j*24+i+2], color=plt.cm.cool(255*i/n), alpha=0.5)
            
            # Plot the estimated transitions
            for i in range(n-1):
                predicted1=learning_algo.transition.predict([abs_states[i:i+1],np.array([[1,0,0,0]])])
                predicted2=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,1,0,0]])])
                predicted3=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,0,1,0]])])
                predicted4=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,0,0,1]])])
                if(self.intern_dim==2):
                    ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), color="0.9", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), color="0.65", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), color="0.4", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), color="0.15", alpha=0.75)
                else:
                    ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.9", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.65", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), np.concatenate([z[i:i+1],predicted3[0,2:3]]), color="0.4", alpha=0.75)
                    ax.plot(np.concatenate([x[i:i+1],predicted4[0,:1]]), np.concatenate([y[i:i+1],predicted4[0,1:2]]), np.concatenate([z[i:i+1],predicted4[0,2:3]]), color="0.15", alpha=0.75)
            
#            for xx in np.arange(self._size_maze)-self._size_maze//2:
#                for yy in np.arange(self._size_maze)-self._size_maze//2:
#                    for zz in np.arange(self._size_maze)-self._size_maze//2:
#                        predicted1=learning_algo.transition.predict([np.array([[xx,yy,zz]]),np.array([[1,0]])])
#                        predicted2=learning_algo.transition.predict([np.array([[xx,yy,zz]]),np.array([[0,1]])])
#                        ax.plot(np.concatenate([np.array([xx]),predicted1[0,:1]]), np.concatenate([np.array([yy]),predicted1[0,1:2]]), np.concatenate([np.array([zz]),predicted1[0,2:]]), color="1", alpha=0.5)
#                        ax.plot(np.concatenate([np.array([xx]),predicted2[0,:1]]), np.concatenate([np.array([yy]),predicted2[0,1:2]]), np.concatenate([np.array([zz]),predicted2[0,2:]]), color="0.5", alpha=0.5)
            
            
            ## Plot the colorbar for the trajectory
            #fig.subplots_adjust(right=0.7)
            #ax1 = fig.add_axes([0.725, 0.15, 0.025, 0.7])
            ## Set the colormap and norm to correspond to the data for which the colorbar will be used.
            #cmap = matplotlib.cm.cool
            #norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
            #
            ## ColorbarBase derives from ScalarMappable and puts a colorbar in a specified axes, so it has 
            ## everything needed for a standalone colorbar.  There are many more kwargs, but the
            ## following gives a basic continuous colorbar with ticks and labels.
            #cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
            #                        norm=norm,
            #                        orientation='vertical')
            #cb1.set_label('Beginning to end of trajectory')
            
            
            # Plot the dots at each time step depending on the action taken
            length_block=[[0,18],[18,19],[19,31]]
            for i in range(3):
                if(self.intern_dim==2):
                    line3 = ax.scatter(all_possib_abs_states[length_block[i][0]:length_block[i][1],0], all_possib_abs_states[length_block[i][0]:length_block[i][1],1], s=30, marker='x', edgecolors='k', alpha=0.5)
                else:
                    line3 = ax.scatter(all_possib_abs_states[length_block[i][0]:length_block[i][1],0], all_possib_abs_states[length_block[i][0]:length_block[i][1],1] ,all_possib_abs_states[length_block[i][0]:length_block[i][1],2], s=30, marker='x', depthshade=True, edgecolors='k', alpha=0.5)
            #line2 = ax.scatter(x, y ,z , c=np.tile(np.expand_dims(1-actions/4.,axis=1),(1,3))-0.125, s=50, marker='o', edgecolors='k', alpha=0.75, depthshade=True)
            #line2 = ax.scatter(x, y, c=np.tile(np.expand_dims(1-actions/4.,axis=1),(1,3))-0.125, s=50, marker='o', edgecolors='k', alpha=0.75)
            if(self.intern_dim==2):
                axes_lims=[ax.get_xlim(),ax.get_ylim()]
            else:
                axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
            
            #zrange=axes_lims[2][1]-axes_lims[2][0]
            
            # Plot the legend for the dots
            from matplotlib.patches import Circle, Rectangle
            from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
#            box1 = TextArea(" State representation (action 0, action 1): ", textprops=dict(color="k"))
#            
#            box2 = DrawingArea(80, 20, 0, 0)
#            el1 = Circle((10, 10), 5, fc="0.9", edgecolor="k", alpha=0.75)
#            el2 = Circle((25, 10), 5, fc="0.65", edgecolor="k", alpha=0.75)
#            el3 = Circle((40, 10), 5, fc="0.4", edgecolor="k", alpha=0.75)
#            el4 = Circle((55, 10), 5, fc="0.15", edgecolor="k", alpha=0.75) 
#            #el3 = Circle((50, 10), 5, fc="0", edgecolor="k") 
#            box2.add_artist(el1)
#            box2.add_artist(el2)
#            box2.add_artist(el3)
#            box2.add_artist(el4)
#           
#           
#            box = HPacker(children=[box1, box2],
#                          align="center",
#                          pad=0, sep=5)
#            
#            anchored_box = AnchoredOffsetbox(loc=3,
#                                             child=box, pad=0.,
#                                             frameon=True,
#                                             bbox_to_anchor=(0., 1.07),
#                                             bbox_transform=ax.transAxes,
#                                             borderpad=0.,
#                                             )
#            ax.add_artist(anchored_box)
            
            
            # Plot the legend for transition estimates
            box1b = TextArea(" Estimated transitions (action 0, 1, 2 and 3): ", textprops=dict(color="k"))
            box2b = DrawingArea(90, 20, 0, 0)
            el1b = Rectangle((5, 10), 15,2, fc="0.9", alpha=0.75)
            el2b = Rectangle((25, 10), 15,2, fc="0.65", alpha=0.75) 
            el3b = Rectangle((45, 10), 15,2, fc="0.4", alpha=0.75)
            el4b = Rectangle((65, 10), 15,2, fc="0.15", alpha=0.75) 
            box2b.add_artist(el1b)
            box2b.add_artist(el2b)
            box2b.add_artist(el3b)
            box2b.add_artist(el4b)
            
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
            
            
            
            #ax.w_xaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
            #ax.w_yaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
            #ax.w_zaxis.set_pane_color((0.99, 0.99, 0.99, 0.99))
            #plt.show()
            plt.savefig('fig_base'+str(learning_algo.update_counter)+'.pdf')


#        # Plot the Q_vals
#        c = learning_algo.Q.predict(np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1))
#        #print "actions,C"
#        #print actions
#        #print c
#        #c=np.max(c,axis=1)
#        m1=ax.scatter(x, y, z+zrange/20, c=c[:,0], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
#        m2=ax.scatter(x, y, z+3*zrange/40, c=c[:,1], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
#        
#        #plt.colorbar(m3)
#        ax2 = fig.add_axes([0.85, 0.15, 0.025, 0.7])
#        cmap = matplotlib.cm.RdYlGn
#        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
#
#        # ColorbarBase derives from ScalarMappable and puts a colorbar
#        # in a specified axes, so it has everything needed for a
#        # standalone colorbar.  There are many more kwargs, but the
#        # following gives a basic continuous colorbar with ticks
#        # and labels.
#        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
#        cb1.set_label('Estimated expected return')
#
#        #plt.show()
#        plt.savefig('fig_w_V'+str(learning_algo.update_counter)+'.pdf')
#
#
#        # fig_visuV
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        
#        x = np.array([i for i in range(5) for jk in range(25)])/4.*(axes_lims[0][1]-axes_lims[0][0])+axes_lims[0][0]
#        y = np.array([j for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[1][1]-axes_lims[1][0])+axes_lims[1][0]
#        z = np.array([k for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[2][1]-axes_lims[2][0])+axes_lims[2][0]
#
#        c = learning_algo.Q.predict(np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1))
#        c=np.max(c,axis=1)
#        #print "c"
#        #print c
#        
#        m=ax.scatter(x, y, z, c=c, vmin=-1., vmax=1., cmap=plt.hot())
#        #plt.colorbar(m)
#        fig.subplots_adjust(right=0.8)
#        ax2 = fig.add_axes([0.875, 0.15, 0.025, 0.7])
#        cmap = matplotlib.cm.hot
#        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
#
#        # ColorbarBase derives from ScalarMappable and puts a colorbar
#        # in a specified axes, so it has everything needed for a
#        # standalone colorbar.  There are many more kwargs, but the
#        # following gives a basic continuous colorbar with ticks
#        # and labels.
#        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
#        cb1.set_label('Estimated expected return')
#
#        #plt.show()
#        plt.savefig('fig_visuV'+str(learning_algo.update_counter)+'.pdf')
#
#
#        # fig_visuR
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')
#        
#        x = np.array([i for i in range(5) for jk in range(25)])/4.*(axes_lims[0][1]-axes_lims[0][0])+axes_lims[0][0]
#        y = np.array([j for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[1][1]-axes_lims[1][0])+axes_lims[1][0]
#        z = np.array([k for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[2][1]-axes_lims[2][0])+axes_lims[2][0]
#
#        coords=np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1)
#        repeat_nactions_coord=np.repeat(coords,self.nActions(),axis=0)
#        identity_matrix = np.diag(np.ones(self.nActions()))
#        tile_identity_matrix=np.tile(identity_matrix,(5*5*5,1))
#
#        c = learning_algo.R.predict([repeat_nactions_coord,tile_identity_matrix])
#        c=np.max(np.reshape(c,(125,self.nActions())),axis=1)
#        #print "c"
#        #print c
#        #mini=np.min(c)
#        #maxi=np.max(c)
#        
#        m=ax.scatter(x, y, z, c=c, vmin=-1., vmax=1., cmap=plt.hot())
#        #plt.colorbar(m)
#        fig.subplots_adjust(right=0.8)
#        ax2 = fig.add_axes([0.875, 0.15, 0.025, 0.7])
#        cmap = matplotlib.cm.hot
#        norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
#
#        # ColorbarBase derives from ScalarMappable and puts a colorbar
#        # in a specified axes, so it has everything needed for a
#        # standalone colorbar.  There are many more kwargs, but the
#        # following gives a basic continuous colorbar with ticks
#        # and labels.
#        cb1 = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap,norm=norm,orientation='vertical')
#        cb1.set_label('Estimated expected return')

        #plt.show()
        plt.savefig('fig_visuR'+str(learning_algo.update_counter)+'.pdf')

        matplotlib.pyplot.close("all") # avoids memory leaks

    def inputDimensions(self):
        if(self._higher_dim_obs==True):
            return [(1,self._size_maze*6,self._size_maze*6)]
        else:
            return [(1,self._size_maze,self._size_maze)]

    def observationType(self, subject):
        return np.float

    def nActions(self):
        return len(self._actions)

    def observe(self):
        obs=copy.deepcopy(self._map)
                
        obs[self._pos_agent[0],self._pos_agent[1]]=0.5                
        if(self._higher_dim_obs==True):
            "self._pos_agent"
            self._pos_agent
            obs=self.get_higher_dim_obs([self._pos_agent],[self._pos_goal])
            
        return [obs]
    
    def get_higher_dim_obs(self,indices_agent,indices_reward):
        obs=copy.deepcopy(self._map)
        obs=obs/1.
        obs=np.repeat(np.repeat(obs, 6, axis=0),6, axis=1)
        # agent repr
        agent_obs=np.zeros((6,6))
        agent_obs[0,2]=0.7
        agent_obs[1,0:5]=0.8
        agent_obs[2,1:4]=0.8
        agent_obs[3,1:4]=0.8
        agent_obs[4,1]=0.8
        agent_obs[4,3]=0.8
        agent_obs[5,0:2]=0.8
        agent_obs[5,3:5]=0.8
        
        # reward repr
        reward_obs=np.zeros((6,6))
        #reward_obs[:,1]=0.8
        #reward_obs[0,1:4]=0.7
        #reward_obs[1,3]=0.8
        #reward_obs[2,1:4]=0.7
        #reward_obs[4,2]=0.8
        #reward_obs[5,2:4]=0.8
        
        for i in indices_reward:
            #print self._map[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6]
            obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6]=reward_obs

        print indices_agent
        for i in indices_agent:
            print i
            obs[i[0]*6:(i[0]+1)*6:,i[1]*6:(i[1]+1)*6]=agent_obs
            
        #plt.imshow(obs, cmap='gray_r')
        #plt.show()
        return obs


    def inTerminalState(self):
#        if((self._pos_agent[0]<=1 and self._cur_action==0) ):#((self._pos_agent==[4,1] and self._cur_action==1) or (self._pos_agent==[5,2] and (self._cur_action==1 or self._cur_action==2)) or (self._pos_agent==[6,3] and self._cur_action==2))):
#        #(self._pos_agent[1]>=self._size_maze-2 and self._cur_action==1) ):
#            return True
        return False
        #if (self._pos_agent==self._pos_goal):
        #    return True
        #else:
        #    return False



if __name__ == "__main__":
    pass
