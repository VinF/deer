"""
Code for general deep Q-learning that can take as inputs scalars, vectors and matrices

.. Authors: Vincent Francois-Lavet, David Taralla

.. Inspired from "Human-level control through deep reinforcement learning",
.. Nature, 518(7540):529-533, February 2015
"""

import numpy as np
import theano
import theano.tensor as T
from .updates import deepmind_rmsprop
from ..base_classes import QNetwork
from .NN_theano import NN # Default Neural network used
    
class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Theano
    
    Parameters
    -----------
    environment : object from class Environment
    rho : float
    rms_epsilon : float
    momentum : float
    clip_delta : float
    freeze_interval : int
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    network_type : str
    update_rule: str
    batch_accumulator : str
    randomState : numpy random number generator
    DoubleQ : bool, optional
        Activate or not the DoubleQ learning, default : False.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    TheQNet : object, optional
        default is deer.qnetworks.NN_theano
    """

    def __init__(self, environment, rho, rms_epsilon, momentum, clip_delta, freeze_interval, batch_size, network_type, 
                 update_rule, batch_accumulator, randomState, DoubleQ=False, TheQNet=NN):
        """ Initialize environment
        
        """
        QNetwork.__init__(self,environment, batch_size)

        
        self.rho = rho
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self._DoubleQ = DoubleQ
        self._randomState = randomState
        
        QNet=TheQNet(self._batch_size, self._input_dimensions, self._n_actions, self._randomState)

        self.update_counter = 0
        
        states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1

        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                states.append(T.tensor4("%s_%s" % ("state", i)))
                next_states.append(T.tensor4("%s_%s" % ("next_state", i)))

            elif len(dim) == 2:
                states.append(T.tensor3("%s_%s" % ("state", i)))
                next_states.append(T.tensor3("%s_%s" % ("next_state", i)))
                
            elif len(dim) == 1:            
                states.append( T.matrix("%s_%s" % ("state", i)) )
                next_states.append( T.matrix("%s_%s" % ("next_state", i)) )
                
            self.states_shared.append(theano.shared(np.zeros((batch_size,) + dim, dtype=theano.config.floatX) , borrow=False))
            self.next_states_shared.append(theano.shared(np.zeros((batch_size,) + dim, dtype=theano.config.floatX) , borrow=False))
        
        print("Number of observations per state: {}".format(len(self.states_shared)))
        print("For each observation, historySize + ponctualObs_i.shape: {}".format(self._input_dimensions))
                
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        thediscount = T.scalar(name='thediscount', dtype=theano.config.floatX)
        thelr = T.scalar(name='thelr', dtype=theano.config.floatX)
        
        QNet=TheQNet(self._batch_size, self._input_dimensions, self._n_actions, self._randomState)
        self.q_vals, self.params, shape_after_conv = QNet._buildDQN(states)
        
        print("Number of neurons after spatial and temporal convolution layers: {}".format(shape_after_conv))

        self.next_q_vals, self.next_params, shape_after_conv = QNet._buildDQN(next_states)
        self._resetQHat()

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        
        
        if(self._DoubleQ==True):
            givens_next={}
            for i, x in enumerate(self.next_states_shared):
                givens_next[ states[i] ] = x

            self.next_q_vals_current_qnet=theano.function([], self.q_vals,
                                          givens=givens_next)

            next_q_curr_qnet = theano.clone(self.next_q_vals)

            argmax_next_q_vals=T.argmax(next_q_curr_qnet, axis=1, keepdims=True)

            max_next_q_vals=self.next_q_vals[T.arange(batch_size),argmax_next_q_vals.reshape((-1,))].reshape((-1, 1))


        else:
            max_next_q_vals=T.max(self.next_q_vals, axis=1, keepdims=True)


        T_ones_like=T.ones_like(T.ones_like(terminals) - terminals)

        target = rewards + T_ones_like * thediscount * max_next_q_vals

        q_val=self.q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))
        # Note : Strangely (target - q_val) lead to problems with python 3.5, theano 0.8.0rc and floatX=float32...
        diff = - q_val + target 

        if self.clip_delta > 0:
            # This loss function implementation is taken from
            # https://github.com/spragunr/deep_q_rl
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            # 
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss_ind = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss_ind = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss_ind)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss_ind)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        givens = {
            rewards: self.rewards_shared,
            actions: self.actions_shared, ## actions not needed!
            terminals: self.terminals_shared
        }
        
        for i, x in enumerate(self.states_shared):
            givens[ states[i] ] = x 
        for i, x in enumerate(self.next_states_shared):
            givens[ next_states[i] ] = x
                
                
        gparams=[]
        for p in self.params:
            gparam =  T.grad(loss, p)
            gparams.append(gparam)

        updates = []
        
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, self.params, gparams, thelr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            for i,(p, g) in enumerate(zip(self.params, gparams)):                
                acc = theano.shared(p.get_value() * 0.)
                acc_new = rho * acc + (1 - self.rho) * g ** 2
                gradient_scaling = T.sqrt(acc_new + self.rms_epsilon)
                g = g / gradient_scaling
                updates.append((acc, acc_new))
                updates.append((p, p - thelr * g))

        elif update_rule == 'sgd':
            for i, (param, gparam) in enumerate(zip(self.params, gparams)):
                updates.append((param, param - thelr * gparam))
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
    
        
        if(self._DoubleQ==True):
            self._train = theano.function([thediscount, thelr, next_q_curr_qnet], [loss, loss_ind, self.q_vals], updates=updates,
                                      givens=givens,
                                      on_unused_input='warn')
        else:
            self._train = theano.function([thediscount, thelr], [loss, loss_ind, self.q_vals], updates=updates,
                                      givens=givens,
                                      on_unused_input='warn')
        givens2={}
        for i, x in enumerate(self.states_shared):
            givens2[ states[i] ] = x 

        self._q_vals = theano.function([], self.q_vals,
                                      givens=givens2,
                                      on_unused_input='warn')

            
    def toDump(self):
        # FIXME

        return None,None
    
    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.

        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training

        Parameters
        -----------
        states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val : b x 1 numpy array of integers
        rewards_val : b x 1 numpy array
        next_states_val : list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val : b x 1 numpy boolean array (currently ignored)


        Returns
        -------
        average loss of the batch training
        """
        
        for i in range(len(self.states_shared)):
            self.states_shared[i].set_value(states_val[i])
            
        for i in range(len(self.states_shared)):
            self.next_states_shared[i].set_value(next_states_val[i])

        
        self.actions_shared.set_value(actions_val.reshape(len(actions_val), 1))
        self.rewards_shared.set_value(rewards_val.reshape(len(rewards_val), 1))
        self.terminals_shared.set_value(terminals_val.reshape(len(terminals_val), 1))
        if self.update_counter % self.freeze_interval == 0:
            self._resetQHat()
        
        if(self._DoubleQ==True):
            self._next_q_curr_qnet = self.next_q_vals_current_qnet()
            loss, loss_ind, _ = self._train(self._df, self._lr,self._next_q_curr_qnet)
        else:
            loss, loss_ind, _ = self._train(self._df, self._lr)

        self.update_counter += 1
        return np.sqrt(loss),loss_ind

    def qValues(self, state_val):
        """ Get the q value for one belief state

        Arguments
        ---------
        state_val : one belief state

        Returns
        -------
        The q value for the provided belief state
        """ 
        # Set the first element of the batch to values provided by state_val
        for i in range(len(self.states_shared)):
            aa = self.states_shared[i].get_value()
            aa[0] = state_val[i]
            self.states_shared[i].set_value(aa)
        
        return self._q_vals()[0]

    def chooseBestAction(self, state):
        """ Get the best action for a belief state

        Arguments
        ---------
        state : one belief state

        Returns
        -------
        The best action : int
        """        
        q_vals = self.qValues(state)

        return np.argmax(q_vals)
        
    def _build(self, network_type, inputs):
        if network_type == "General_DQN_0":
            return self._buildDQN(inputs)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def _resetQHat(self):
        
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            next_param.set_value(param.get_value())        

