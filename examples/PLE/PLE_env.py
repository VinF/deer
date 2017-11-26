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
        
        return [2 * [48 * [48 * [0]]]]
        
        
    def act(self, action):
        action = self._actions[action]
        
        #if self._mode == MyEnv.VALIDATION_MODE:
        #    action=0

        reward = 0
        for _ in range(self._frame_skip):
            reward += self._ple.act(action)
            if self.inTerminalState():
                break
            
        self._screen = self._ple.getScreenGrayscale()
        cv2.resize(self._screen, (48, 48), self._reduced_screen, interpolation=cv2.INTER_NEAREST)
  
        self._mode_score += reward
        return np.sign(reward)

    def summarizePerformance(self, test_data_set, learning_algo):
        #print "test_data_set.observations.shape"
        #print test_data_set.observations()[0][0:1]
        n=20
        historics=[]
        for i,observ in enumerate(test_data_set.observations()[0][0:n]):
            if(i<n-1):
                historics.append(np.expand_dims(observ,axis=0))
            if(i>0):
                historics[i-1]=np.concatenate([historics[i-1],np.expand_dims(observ,axis=0)], axis=0)
        historics=np.array(historics)
        #print historics
        abs_states=learning_algo.encoder.predict(historics)
        print abs_states
        actions=test_data_set.actions()[0:n]
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
        
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        for i in xrange(n-1):
            ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=plt.cm.jet(255*i/n))
        #line = ax.contour(x, y ,z, cmap=cm.coolwarm)
        line2 = ax.scatter(x, y ,z , c=np.tile(np.expand_dims(actions/2.,axis=1),(1,3)), s=50, marker='o', edgecolors='none', depthshade=False)
        #m.set_array(actions/2.)
        #plt.colorbar(m)
                
        #plt.show()
        plt.savefig('fig'+str(learning_algo.update_counter)+'.pdf')


    def inputDimensions(self):
        return [(2, 48, 48)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        return [np.array(self._reduced_screen)/256.]

    def inTerminalState(self):
        return self._ple.game_over()
                


if __name__ == "__main__":
    pass
