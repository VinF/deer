"""
Code for the actor-critic "DDPG" (https://arxiv.org/abs/1509.02971)

.. Author: Vincent Francois-Lavet
"""

import sys
import numpy as np
from ..base_classes import QNetwork as ACNetwork
from .NN_keras import NN # Default Neural network used
from warnings import warn
from keras.optimizers import SGD,RMSprop
from keras import backend as K

try:
    import tensorflow as tf
    assert(K.backend()=="tensorflow")
except:
    print('Error : Currently only Tensorflow is supported as a backend for AC_net_keras. You can make the switch in the file ~/.keras/keras.json')
    #sys.exit(0)

class MyACNetwork(ACNetwork):
    """
    Actor-critic learning (using Keras) with Deep Deterministic Policy Gradient (DDPG) for the continuous action domain
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Default : 0
    clip_delta : float
        Not implemented.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    batch_accumulator : str
        {sum,mean}. Default : sum
    random_state : numpy random number generator
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        default is deer.qnetworks.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_delta=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network_critic=NN, neural_network_actor=NN):
        """ Initialize environment
        
        """
        ACNetwork.__init__(self,environment, batch_size)

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0
        
        self.sess = tf.Session()
        K.set_session(self.sess)
        
        Q_net = neural_network_critic(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, True)
        
        self.q_vals, self.params, self.inputsQ = Q_net._buildDQN()
        
        if (update_rule=="sgd"):
            optimizer = SGD(lr=self._lr, momentum=self._momentum, nesterov=False)
        elif (update_rule=="rmsprop"):
            optimizer = RMSprop(lr=self._lr, rho=self._rho, epsilon=self._rms_epsilon)
        else:
            raise Exception('The update_rule '+update_rule+ 'is not implemented.')
        
        self.q_vals.compile(optimizer=optimizer, loss='mse')
       
        self.next_q_vals, self.next_params, self.next_inputsQ = Q_net._buildDQN()
        self.next_q_vals.compile(optimizer='rmsprop', loss='mse') #The parameters do not matter since training is done on self.q_vals

        self._resetQHat()
        

        policy_net = neural_network_actor(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, False)
        self.policy, self.params_policy = policy_net._buildDQN()
        self.policy.compile(optimizer=optimizer, loss='mse')
        self.next_policy, self.next_params_policy = policy_net._buildDQN()
        self.next_policy.compile(optimizer=optimizer, loss='mse')
        
        
        
        ### self.policy
        self.action_grads = tf.gradients(self.q_vals.output,self.inputsQ[-1])  #GRADIENTS for policy update
       
        
        self.sess.run(tf.initialize_all_variables())        


    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(K.get_value(p))
        for i,p in enumerate(self.params_policy):
            params_value.append(K.get_value(p))
        
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            K.set_value(p,list_of_values[i])
        for j,p in enumerate(self.params_policy):
            K.set_value(p,list_of_values[j+i+1])

    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of objects (lists of floats)
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array (currently ignored)


        Returns
        -------
        Average loss of the batch training
        Individual losses for each tuple
        """
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        

        ### Tain self.q_vals
        next_actions_val=self.next_policy.predict(next_states_val.tolist())

        ns_list=next_states_val.tolist()
        ns_list.append( next_actions_val )
        next_q_vals = self.next_q_vals.predict(  ns_list  )
        
        not_terminals=np.ones_like(terminals_val) - terminals_val
        
        target = rewards_val + not_terminals * self._df * next_q_vals.reshape((-1))
        
        s_list=states_val.tolist()
        s_list.append( actions_val  )
        
        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        q_vals=self.q_vals.predict( s_list ).reshape((-1))
        diff_q = - q_vals + target 
        loss_ind_q=pow(diff_q,2)
        
        loss_q=self.q_vals.train_on_batch( s_list , target ) 
        
        
        ### Tain self.policy
        cur_action=self.policy.predict(states_val.tolist())
        cur_action=self.clip_action(cur_action)
        gg=self.gradients(states_val.tolist(),cur_action)
        
        target_action=self.clip_action(cur_action+gg)
        
        # In order to obtain the individual losses, we predict the current Q_vals and calculate the diff
        diff_policy = - cur_action + target_action
        loss_ind_policy=np.sum(pow(diff_policy,2),axis=-1)

        loss_policy=self.policy.train_on_batch(states_val.tolist(), target_action)
                        
        self.update_counter += 1        
        
        
        return loss_q+loss_policy,loss_ind_q+loss_ind_policy


    def clip_action(self, action):
        return np.clip(action,-1,1) #FIXME
    

    def gradients(self, states, actions):
        feed_dict={}
        for i,s in enumerate(states):
            #print i,s
            feed_dict[ self.inputsQ[i] ] = s
        
        feed_dict[ self.inputsQ[-1] ] = actions#np.expand_dims(actions,1)
        
        out=self.sess.run(self.action_grads, feed_dict=feed_dict)[0]
        
        return out

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        best_action : float
        estim_value : float
        """        
        
        best_action=self.policy.predict([np.expand_dims(s,axis=0) for s in state])
        the_list=[np.expand_dims(s,axis=0) for s in state]
        the_list.append( best_action )
        estim_value=(self.q_vals.predict(the_list)[0,0])
        
        return best_action[0],estim_value
        
    def _resetQHat(self):
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            K.set_value(next_param,K.get_value(param))
