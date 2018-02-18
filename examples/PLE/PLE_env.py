""" Interface with the PLE environment

Authors: Vincent Francois-Lavet, David Taralla
Modified by: Norman Tasfi
"""
import numpy as np
import cv2
from ple import PLE

from deer.base_classes import Environment

import matplotlib
matplotlib.use('qt5agg')
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.pyplot as plt

class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, game=None, frame_skip=2, 
            ple_options={"display_screen": True, "force_fps":True, "fps":15}):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._frame_skip = frame_skip if frame_skip >= 1 else 1
        self._random_state = rng
       
        if game is None:
            raise ValueError("Game must be provided")

        self._ple = PLE(game, **ple_options)
        self._ple.game.rng = rng
        self._ple.init()

        w, h = self._ple.getScreenDims()
        self._screen = np.empty((h, w), dtype=np.uint8)
        self._reduced_screen = np.empty((48, 48), dtype=np.uint8)
        self._actions = self._ple.getActionSet()

                
    def reset(self, mode):
        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                # fix the seed for every new validation. It potentially removes one source of variance and
                # it allows to show some illustration of the learning for the same setting in validation
                self._ple.game.rng = np.random.RandomState(23) # 23:left, center, right, ...
            else:
                self._mode_episode_count += 1
        elif self._mode != -1: # and thus mode == -1
            self._mode = -1
        
        
        self._ple.reset_game()
        #for _ in range(self._ple.rng.randint(15)):
        #    self._ple.act(self._ple.NOOP)
        self._screen = self._ple.getScreenGrayscale()
        cv2.resize(self._screen, (48, 48), self._reduced_screen, interpolation=cv2.INTER_NEAREST)
        
        return [1 * [48 * [48 * [0]]]]
        
        
    def act(self, action):
        #print action
        #print self._actions
        #if self._mode == MyEnv.VALIDATION_MODE:
        #    action=0
        action = self._actions[action]
        

        self.reward = 0
        for _ in range(self._frame_skip):
            self.reward += self._ple.act(action)
            if self.inTerminalState():
                break
            
        self._screen = self._ple.getScreenGrayscale()
        cv2.resize(self._screen, (48, 48), self._reduced_screen, interpolation=cv2.INTER_NEAREST)
  
        self._mode_score += self.reward
        return np.sign(self.reward)

    def summarizePerformance(self, test_data_set, learning_algo):
        #print "test_data_set.observations.shape"
        #print test_data_set.observations()[0][0:1]
        n=14
        historics=[]
        for i,observ in enumerate(test_data_set.observations()[0][0:n]):
            historics.append(np.expand_dims(observ,axis=0))
#        for i,observ in enumerate(test_data_set.observations()[0][0:n+1]):
#            if(i<n):
#                historics.append(np.expand_dims(observ,axis=0))
#            if(i>0):
#                historics[i-1]=np.concatenate([historics[i-1],np.expand_dims(observ,axis=0)], axis=0)
        historics=np.array(historics)
        #print historics
        abs_states=learning_algo.encoder.predict(historics)
        print abs_states
        actions=test_data_set.actions()[0:n] #instead of 0:n because history of 2 time steps considered
        print actions
        print test_data_set.rewards()[0:n]
        if self.inTerminalState() == False:
            self._mode_episode_count += 1
        print("== Mean score per episode is {} over {} episodes ==".format(self._mode_score / self._mode_episode_count, self._mode_episode_count))
                
        
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
        
        # Plot the trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        for i in xrange(n-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.cool(255*i/n), alpha=0.5)

        # Plot the fitted one-step trajectory from time t=10
        for i in range(n-1):
            predicted1=learning_algo.transition.predict([abs_states[i:i+1],np.array([[1,0,0]])])
            predicted2=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,1,0]])])
            predicted3=learning_algo.transition.predict([abs_states[i:i+1],np.array([[0,0,1]])])
            ax.plot(np.concatenate([x[i:i+1],predicted3[0,:1]]), np.concatenate([y[i:i+1],predicted3[0,1:2]]), np.concatenate([z[i:i+1],predicted3[0,2:3]]), color="0.23", alpha=0.75) #black
            ax.plot(np.concatenate([x[i:i+1],predicted2[0,:1]]), np.concatenate([y[i:i+1],predicted2[0,1:2]]), np.concatenate([z[i:i+1],predicted2[0,2:3]]), color="0.57", alpha=0.75) #grey
            ax.plot(np.concatenate([x[i:i+1],predicted1[0,:1]]), np.concatenate([y[i:i+1],predicted1[0,1:2]]), np.concatenate([z[i:i+1],predicted1[0,2:3]]), color="0.9", alpha=0.75) #white

#        for xx in [-2,-1.,0, 1., 2.]:
#            for yy in [-2,-1.,0, 1., 2.]:
#                for zz in [-2,-1.,0, 1., 2.]:
#                    predicted1=learning_algo.transition2.predict([np.array([[xx,yy,zz]]),np.array([[1,0,0]])])
#                    predicted2=learning_algo.transition2.predict([np.array([[xx,yy,zz]]),np.array([[0,1,0]])])
#                    predicted3=learning_algo.transition2.predict([np.array([[xx,yy,zz]]),np.array([[0,0,1]])])
#                    ax.plot(np.concatenate([np.array([xx]),predicted1[0,:1]]), np.concatenate([np.array([yy]),predicted1[0,1:2]]), np.concatenate([np.array([zz]),predicted1[0,2:]]), color="1", alpha=0.5)
#                    ax.plot(np.concatenate([np.array([xx]),predicted2[0,:1]]), np.concatenate([np.array([yy]),predicted2[0,1:2]]), np.concatenate([np.array([zz]),predicted2[0,2:]]), color="0.5", alpha=0.5)
#                    ax.plot(np.concatenate([np.array([xx]),predicted3[0,:1]]), np.concatenate([np.array([yy]),predicted3[0,1:2]]), np.concatenate([np.array([zz]),predicted3[0,2:]]), color="0", alpha=0.5)
                    #ax.plot(np.concatenate([x[i:i+1],predicted[0,:1]]), np.concatenate([y[i:i+1],predicted[0,1:2]]), np.concatenate([z[i:i+1],predicted[0,2:]]), color="g")
        

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
        print np.tile(np.expand_dims(actions,axis=1),(1,3))
        print np.tile(np.expand_dims(0.75-actions/4.,axis=1),(1,3))
        line2 = ax.scatter(x, y ,z , c=np.tile(np.expand_dims(0.9-actions/3.,axis=1),(1,3)), s=50, marker='o', edgecolors='k', depthshade=True, alpha=0.75)
        axes_lims=[ax.get_xlim(),ax.get_ylim(),ax.get_zlim()]
        zrange=axes_lims[2][1]-axes_lims[2][0]
        
        # Plot the legend for the dots
        from matplotlib.patches import Circle, Rectangle
        from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, DrawingArea, HPacker
        box1 = TextArea(" State representation (action 0, 1 or 2) : ", textprops=dict(color="k")) #none, left and right
        
        box2 = DrawingArea(60, 20, 0, 0)
        el1 = Circle((10, 10), 5, fc="0.9", alpha=0.75, edgecolor="k")
        el2 = Circle((30, 10), 5, fc="0.57", alpha=0.75, edgecolor="k") 
        el3 = Circle((50, 10), 5, fc="0.23", alpha=0.75, edgecolor="k") 
        box2.add_artist(el1)
        box2.add_artist(el2)
        box2.add_artist(el3)
        
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
        box1b = TextArea(" Estimated transitions (action 0, 1 or 2): ", textprops=dict(color="k"))
        box2b = DrawingArea(70, 20, 0, 0)
        el1b = Rectangle((5, 10), 15,2, fc="0.9", alpha=0.75)
        el2b = Rectangle((25, 10), 15,2, fc="0.57", alpha=0.75) 
        el3b = Rectangle((45, 10), 15,2, fc="0.23", alpha=0.75) 
        box2b.add_artist(el1b)
        box2b.add_artist(el2b)
        box2b.add_artist(el3b)

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
        plt.savefig('fig_base'+str(learning_algo.update_counter)+'.pdf')


        # Plot the Q_vals
        c = learning_algo.Q.predict(np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1))
        #print "actions,C"
        #print actions
        #print c
        #c=np.max(c,axis=1)
        m1=ax.scatter(x, y, z+zrange/20, c=c[:,0], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
        m2=ax.scatter(x, y, z+3*zrange/40, c=c[:,1], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
        m3=ax.scatter(x, y, z+zrange/10, c=c[:,2], vmin=-1., vmax=1., cmap=plt.cm.RdYlGn)
        
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

        plt.savefig('fig_w_V'+str(learning_algo.update_counter)+'.pdf')


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
        plt.savefig('fig_visuV'+str(learning_algo.update_counter)+'.pdf')


        # fig_visuR
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.array([i for i in range(5) for jk in range(25)])/4.*(axes_lims[0][1]-axes_lims[0][0])+axes_lims[0][0]
        y = np.array([j for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[1][1]-axes_lims[1][0])+axes_lims[1][0]
        z = np.array([k for i in range(5) for j in range(5) for k in range(5)])/4.*(axes_lims[2][1]-axes_lims[2][0])+axes_lims[2][0]

        coords=np.concatenate((np.expand_dims(x,axis=1),np.expand_dims(y,axis=1),np.expand_dims(z,axis=1)),axis=1)
        repeat3_coord=np.repeat(coords,3,axis=0)
        identity_matrix = np.diag(np.ones(self.nActions()))
        tile_identity_matrix=np.tile(identity_matrix,(5*5*5,1))

        c = learning_algo.R.predict([repeat3_coord,tile_identity_matrix])
        c=np.max(np.reshape(c,(125,3)),axis=1)
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
        plt.savefig('fig_visuR'+str(learning_algo.update_counter)+'.pdf')

        matplotlib.pyplot.close("all") # avoids memory leaks

    def inputDimensions(self):
        return [(1, 48, 48)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        return [np.array(self._reduced_screen)/256.]

    def inTerminalState(self):
        #if (self.reward!=0):
        #    # If a reward has been observed, end the episode
        #    print "end!!"
        #    return True
        #else:
        return self._ple.game_over()
                


if __name__ == "__main__":
    pass
