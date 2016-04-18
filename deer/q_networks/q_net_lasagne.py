"""
Code for general deep Q-learning that can take as inputs scalars, vectors and matrices

Authors: Vincent Francois-Lavet, David Taralla

Inspired from "Human-level control through deep reinforcement learning",
Nature, 518(7540):529-533, February 2015
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T
from .updates import deepmind_rmsprop, get_or_compute_grads
from ..base_classes import QNetwork

class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, environment, rho, rms_epsilon, momentum, clip_delta, freeze_interval, batchSize, network_type, 
                 update_rule, batch_accumulator, randomState, frame_scale=255.0):
        """ Initialize environment

        Arguments:
            environment - the environment (class Env) 
            num_elements_in_batch - list of k integers for the number of each element kept as belief state
            num_actions - int
            discount - float
            learning_rate - float
            rho, rms_epsilon, momentum - float, float, float
            ...
            network_type - string 
            ...           
        """

        self._environment = environment
        
        self._batchSize = batchSize
        self._inputDimensions = self._environment.inputDimensions()
        self._nActions = self._environment.nActions()
        self._df = 0
        self.rho = rho
        self._lr = 0
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self._randomState = randomState
        
        lasagne.random.set_rng(self._randomState)

        self.update_counter = 0
        
        states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1

        for i, dim in enumerate(self._inputDimensions):
            if len(dim) == 3:
                states.append(T.tensor4("%s_%s" % ("state", i)))
                next_states.append(T.tensor4("%s_%s" % ("next_state", i)))

            elif len(dim) == 2:
                states.append(T.tensor3("%s_%s" % ("state", i)))
                next_states.append(T.tensor3("%s_%s" % ("next_state", i)))
                
            elif len(dim) == 1:            
                states.append( T.matrix("%s_%s" % ("state", i)) )
                next_states.append( T.matrix("%s_%s" % ("next_state", i)) )
                
            self.states_shared.append(theano.shared(np.zeros((batchSize,) + dim, dtype=theano.config.floatX) , borrow=False))
            self.next_states_shared.append(theano.shared(np.zeros((batchSize,) + dim, dtype=theano.config.floatX) , borrow=False))
        
        print("Number of observations per state: {}".format(len(self.states_shared)))
        print("For each observation, historySize + ponctualObs_i.shape: {}".format(self._inputDimensions))
                
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        thediscount = T.scalar(name='thediscount', dtype=theano.config.floatX)
        thelr = T.scalar(name='thelr', dtype=theano.config.floatX)
        
        self.l_out, self.l_outs_conv, shape_after_conv = self._build(network_type, states)
        
        print("Number of neurons after spatial and temporal convolution layers: {}".format(shape_after_conv))

        self.next_l_out, self.next_l_outs_conv, shape_after_conv = self._build(network_type, next_states)
        self._resetQHat()

        self.rewards_shared = theano.shared(
            np.zeros((batchSize, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batchSize, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batchSize, 1), dtype='int32'),
            broadcastable=(False, True))


        q_vals = lasagne.layers.get_output(self.l_out)        
        
        next_q_vals = lasagne.layers.get_output(self.next_l_out)
        
        max_next_q_vals=T.max(next_q_vals, axis=1, keepdims=True)
        
        T_ones_like=T.ones_like(T.ones_like(terminals) - terminals)
        
        target = rewards + T_ones_like * thediscount * max_next_q_vals

        q_val=q_vals[T.arange(batchSize), actions.reshape((-1,))].reshape((-1, 1))

        diff = target - q_val

        if self.clip_delta > 0:
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
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.l_out)

        for conv_param in self.l_outs_conv:
            for p in lasagne.layers.helper.get_all_params(conv_param):
                params.append(p)
        
            
        givens = {
            rewards: self.rewards_shared,
            actions: self.actions_shared, ## actions not needed!
            terminals: self.terminals_shared
        }
        
        for i, x in enumerate(self.states_shared):
            givens[ states[i] ] = x 
        for i, x in enumerate(self.next_states_shared):
            givens[ next_states[i] ] = x
                
        if update_rule == 'deepmind_rmsprop':
            grads = get_or_compute_grads(loss, params)
            updates = deepmind_rmsprop(loss, params, grads, thelr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, thelr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, thelr)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([thediscount, thelr], [loss, q_vals], updates=updates,
                                      givens=givens,
                                      on_unused_input='warn')
        givens2={}
        for i, x in enumerate(self.states_shared):
            givens2[ states[i] ] = x 

        self._q_vals = theano.function([], q_vals,
                                      givens=givens2,
                                      on_unused_input='warn')

    def setLearningRate(self, lr):
        self._lr = lr

    def setDiscountFactor(self, df):
        self._df = df

    def learningRate(self):
        return self._lr

    def discountFactor(self):
        return self._df
            
    def toDump(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        all_params_conv=[]
        for conv_param in self.l_outs_conv:
            all_params_conv.append(lasagne.layers.helper.get_all_param_values(conv_param))

        return all_params, all_params_conv
    
    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train one batch.
        1. Set shared variable in states_shared, next_states_shared, actions_shared, rewards_shared, terminals_shared         
        2. perform batch training
        
        Arguments:
        states_val - list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        actions_val - b x 1 numpy array of integers
        rewards_val - b x 1 numpy array
        next_states_val - list of batch_size * [list of max_num_elements* [list of k * [element 2D,1D or scalar]])
        terminals_val - b x 1 numpy boolean array (currently ignored)

        Returns: average loss of the batch training
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
            
        loss, _ = self._train(self._df, self._lr)
        self.update_counter += 1
        return np.sqrt(loss)

    def qValues(self, state_val):
        """ Get the q value for one belief state

        Arguments:
            states_val - list of max_num_elements* [list of k * [element 2D,1D or scalar]]

        Returns:
           The q value for the provided belief state
        """ 
        for i in range(len(self.states_shared)):
            aa = self.states_shared[i].get_value()
            aa[0] = state_val[i]
            self.states_shared[i].set_value(aa)
        
        return self._q_vals()[0]

    def chooseBestAction(self, states):
        """ Get the best action for a batch of states

        Arguments:
            states - list of lists of max_num_elements* [list of k * [element 2D,1D or scalar]]

        Returns:
           The q value for the provided belief state
        """        
        q_vals = self.qValues(states)

        return np.argmax(q_vals)
        
    def _build(self, network_type, inputs):
        if network_type == "General_DQN_0":
            return self._buildG_DQN_0(inputs)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))

    def _resetQHat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        
        all_params_conv=[]
        for conv_param in self.l_outs_conv:
            all_params_conv.append( lasagne.layers.helper.get_all_param_values(conv_param) )

        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)
        for i,param_conv in enumerate(all_params_conv):
            lasagne.layers.helper.set_all_param_values(self.next_l_outs_conv[i], param_conv)        
        

    def _buildG_DQN_0(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        if ("gpu" in theano.config.device):
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
            conv2DFunc = Conv2DCCLayer
        else:
            conv2DFunc = lasagne.layers.Conv2DLayer

        l_outs_conv=[]
        for i, dim in enumerate(self._inputDimensions):
            # - observation[i] is a FRAME -
            if len(dim) == 3: 
                # Building here for 3D
                l_in = lasagne.layers.InputLayer(
                    shape=(self._batchSize,) + dim, 
                    input_var=inputs[i],
                )
                
                l_conv1 = conv2DFunc(
                    l_in,
                    num_filters=32,
                    filter_size=(8, 8),
                    stride=(4, 4),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0)
                )
                
                l_conv2 = conv2DFunc(
                    l_conv1,
                    num_filters=64,
                    filter_size=(4, 4),
                    stride=(2, 2),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0)
                )
                
                l_conv3 = conv2DFunc(
                    l_conv2,
                    num_filters=64,
                    filter_size=(3, 3),
                    stride=(1, 1),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0)
                )
                l_outs_conv.append(l_conv3)
                
            # - observation[i] is a VECTOR -
            elif len(dim) == 2 and dim[0] > 3:
                # Building here for  2D
                l_in = lasagne.layers.InputLayer(
                    shape=(self._batchSize, 1) + dim, 
                    input_var=inputs[i].reshape((self._batchSize, 1) + dim),
                )
                
                l_conv1 = conv2DFunc(
                    l_in,
                    num_filters=16,
                    filter_size=(2, 1),#filter_size=(8, 8),
                    stride=(1, 1),#stride=(4, 4),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0)
                )
                
                l_conv2 = conv2DFunc(
                    l_conv1,
                    num_filters=16,
                    filter_size=(2, 2),
                    stride=(1, 1),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                )
                l_outs_conv.append(l_conv2)
                
            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    # Building here for  1D
                    l_in = lasagne.layers.InputLayer(
                        shape=(self._batchSize, 1) + dim, 
                        input_var=inputs[i].reshape((self._batchSize, 1) + dim),
                    )
                
                    l_conv1 = lasagne.layers.Conv1DLayer(
                        l_in,
                        num_filters=8,#32,
                        filter_size=2,#filter_size=(8, 8),
                        stride=1,#stride=(4, 4),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(), # Defaults to Glorot
                        b=lasagne.init.Constant(.0)
                    )
                
                    l_conv2 = lasagne.layers.Conv1DLayer(
                        l_conv1,
                        num_filters=8,#64,
                        filter_size=2,#filter_size=(4, 4),
                        stride=1,#stride=(2, 2),
                        nonlinearity=lasagne.nonlinearities.rectify,
                        W=lasagne.init.HeUniform(),
                        b=lasagne.init.Constant(.0)
                    )
                
                    l_outs_conv.append(l_conv2)
                else:
                    # Building here for 1D simple
                    l_in = lasagne.layers.InputLayer(
                        shape=(self._batchSize, 1) + dim, 
                        input_var=inputs[i].reshape((self._batchSize, 1) + dim),
                    )
                                
                    l_outs_conv.append(l_in)

        ## Custom merge of layers
        ## NB : l_output_conv=lasagne.layers.MergeLayer(l_outs_conv) gives NOT IMPLEMENTED ERROR
        output_conv = lasagne.layers.get_output(l_outs_conv[0]).flatten().reshape((self._batchSize, np.prod(l_outs_conv[0].output_shape[1:])))       
        shapes = [np.prod(l_outs_conv[0].output_shape[1:])]

        if (len(l_outs_conv)>1):
            for l_out_conv in l_outs_conv[1:]:
                output_conv=T.concatenate((output_conv, lasagne.layers.get_output(l_out_conv).flatten().reshape((self._batchSize, np.prod(l_out_conv.output_shape[1:])))) , axis=1)
                shapes.append(np.prod(l_out_conv.output_shape[1:]))

        shape = sum(shapes)

        l_output_conv = lasagne.layers.InputLayer(
            shape=([self._batchSize, shape]),
            input_var=output_conv,
        )

        l_hidden1 = lasagne.layers.DenseLayer(
            l_output_conv,
            num_units=50,#512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.0)
        )

        l_hidden2 = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=20,#50,#512,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.0)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden2,
            num_units=self._nActions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.0)
        )

        return l_out, l_outs_conv, shapes


if __name__ == '__main__':
    pass
