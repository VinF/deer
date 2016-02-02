"""
Code for general deep Q-learning that can take as inputs scalars, vectors and matrices

Author: Vincent Francois-Lavet

Inspired from "Human-level control through deep reinforcement learning",
Nature, 518(7540):529-533, February 2015 and the implementation of Nathan 
Sprague (https://github.com/spragunr/deep_q_rl)
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T
from updates import deepmind_rmsprop

class DeepQLearner:
    """
    Deep Q-learning network using Lasagne.
    """

    def __init__(self, environment, 
                 num_actions, discount, discount_inc, learning_rate, learning_rate_decay, rho,
                 rms_epsilon, momentum, clip_delta, freeze_interval,
                 batch_size, network_type, update_rule,
                 batch_accumulator, rng, frame_scale=255.0):
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

        self.environment=environment
        
        self.num_elements_in_batch = self.environment.num_elements_in_batch
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.discount = discount
        self.discount_inc = discount_inc
        self.rho = rho
        self.lr = learning_rate
        self.lr_dec = learning_rate_decay
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta
        self.freeze_interval = freeze_interval
        self.rng = rng
        
        lasagne.random.set_rng(self.rng)

        self.update_counter = 0
        
        states=[]   # list of symbolic variables for each of the k element in the belief state
                    # --> [ T.tensor4 if observation of element=matrix, T.tensor3 if vector, T.tensor 2 if scalar ]
        next_states=[] # idem than states at t+1 
        self.states_shared=[] # list of shared variable for each of the k element in the belief state
        self.next_states_shared=[] # idem that self.states_shared at t+1
        shapes=[]

        
        for i,(element,num_element) in enumerate( zip(self.environment.observation, self.num_elements_in_batch) ):
            element=np.array(element)

            if (element.ndim==2):
                states.append( T.tensor4("%s_%s" % ("state", i)) )
                next_states.append( T.tensor4("%s_%s" % ("next_state", i)) )
                
                shapes.append(element.shape)
                self.states_shared.append( theano.shared( np.zeros((batch_size, num_element, element.shape[0], element.shape[1]), dtype=theano.config.floatX) , borrow=False) )
                self.next_states_shared.append( theano.shared( np.zeros((batch_size, num_element, element.shape[0], element.shape[1]), dtype=theano.config.floatX) , borrow=False) )

            if (element.ndim==1):
                
                states.append( T.tensor3("%s_%s" % ("state", i)) )
                next_states.append( T.tensor3("%s_%s" % ("next_state", i)) )
                
                shapes.append(element.shape)
                self.states_shared.append( theano.shared( np.zeros((batch_size, num_element, element.shape[0]), dtype=theano.config.floatX) , borrow=False) )
                self.next_states_shared.append( theano.shared( np.zeros((batch_size, num_element, element.shape[0]), dtype=theano.config.floatX) , borrow=False) )

            if (element.ndim==0):                
                states.append( T.matrix("%s_%s" % ("state", i)) )
                next_states.append( T.matrix("%s_%s" % ("next_state", i)) )

                shapes.append((1,))
                self.states_shared.append( theano.shared( np.zeros((batch_size, num_element), dtype=theano.config.floatX) , borrow=False) )
                self.next_states_shared.append( theano.shared( np.zeros((batch_size, num_element), dtype=theano.config.floatX) , borrow=False) )

        
        print "Number of elements for the state: "+str(len(self.states_shared))
        print "Shape of each of the elements: "+str(shapes)
        print "History to consider for each element: "+str(self.num_elements_in_batch) 
                
        rewards = T.col('rewards')
        actions = T.icol('actions')
        terminals = T.icol('terminals')
        thediscount = T.scalar(name='thediscount', dtype=theano.config.floatX)
        thelr = T.scalar(name='thelr', dtype=theano.config.floatX)
        
        self.l_out, self.l_outs_conv, shape_after_conv = self.build_network(network_type, self.num_elements_in_batch, shapes,
                                        num_actions, batch_size, states)#_shared)
        
        print "Number of neurons after spatial and temporal convolution layers: "+str(shape_after_conv)

        if self.freeze_interval > 0:
            self.next_l_out, self.next_l_outs_conv, shape_after_conv = self.build_network(network_type, self.num_elements_in_batch, shapes,
                                                 num_actions, batch_size, next_states)
            self.reset_q_hat()

        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))

        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        self.terminals_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))


        q_vals = lasagne.layers.get_output(self.l_out)        
        

        if self.freeze_interval > 0:
            next_q_vals = lasagne.layers.get_output(self.next_l_out)
        else:
            next_q_vals = lasagne.layers.get_output(self.l_out)
            next_q_vals = theano.gradient.disconnected_grad(next_q_vals)
        
        max_next_q_vals=T.max(next_q_vals, axis=1, keepdims=True)
        
        T_ones_like=T.ones_like(T.ones_like(terminals) - terminals)
        
        target = rewards + T_ones_like * thediscount * max_next_q_vals

        q_val=q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))

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
        
            
        params.append  
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
            updates = deepmind_rmsprop(loss, params, thelr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, self.lr)
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

    def build_network(self, network_type, num_elements_in_batch, shapes,
                      output_dim, batch_size, inputs):
        if network_type == "General_DQN_0":
            return self.build_my_network(num_elements_in_batch, shapes,
                                             output_dim, batch_size, inputs)
        else:
            raise ValueError("Unrecognized network: {}".format(network_type))
            

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
        
        for i, x in enumerate(self.states_shared):

            swap_states_val=np.array(np.array(states_val)[:,:,i])
            
            for j,swap_states_val1 in enumerate(swap_states_val[0,:]):
                if(swap_states_val1==None):
                    swap_states_val=np.delete(swap_states_val,np.s_[j:],axis=1)                    
                    break
                
            swap_states_val=np.asarray( swap_states_val.tolist() , dtype=theano.config.floatX)
                            
            self.states_shared[i].set_value(swap_states_val)

        for i, x in enumerate(self.next_states_shared):
            swap_states_val=np.array(np.array(next_states_val)[:,:,i])
            
            for j,swap_states_val1 in enumerate(swap_states_val[0,:]):
                if(swap_states_val1==None):
                    swap_states_val=np.delete(swap_states_val,np.s_[j:],axis=1)                    
                    break
                
            swap_states_val=np.asarray( swap_states_val.tolist() , dtype=theano.config.floatX)
                            
            self.next_states_shared[i].set_value( swap_states_val )

        
        self.actions_shared.set_value(actions_val)
        self.rewards_shared.set_value(rewards_val)
        self.terminals_shared.set_value(terminals_val)
        if (self.freeze_interval > 0 and
            self.update_counter % self.freeze_interval == 0):
            self.reset_q_hat()
            
        loss, _ = self._train(self.discount, self.lr)
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state_val):
        """ Get the q value for one belief state

        Arguments:
            states_val - list of max_num_elements* [list of k * [element 2D,1D or scalar]]

        Returns:
           The q value for the provided belief state
        """        
        for i, x in enumerate(self.states_shared):            
            swap_states_val=np.array(np.array(state_val)[:,i])

            for j,swap_states_val1 in enumerate(swap_states_val[:]):
                if(swap_states_val1==None):
                    swap_states_val=np.delete(swap_states_val,np.s_[j:],axis=0)                    
                    break
                
            swap_states_val=np.asarray( swap_states_val.tolist() , dtype=theano.config.floatX)

            aa=self.states_shared[i].get_value()
            aa[0]=swap_states_val
                            
            self.states_shared[i].set_value(aa)
        
        return self._q_vals()[0]

    def choose_best_action(self, states):
        """ Get the best action for a batch of states

        Arguments:
            states - list of lists of max_num_elements* [list of k * [element 2D,1D or scalar]]

        Returns:
           The q value for the provided belief state
        """        
        q_vals = self.q_vals(states)

        return np.argmax(q_vals)

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        
        all_params_conv=[]
        for conv_param in self.l_outs_conv:
            all_params_conv.append( lasagne.layers.helper.get_all_param_values(conv_param) )

        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)
        for i,param_conv in enumerate(all_params_conv):
            lasagne.layers.helper.set_all_param_values(self.next_l_outs_conv[i], param_conv)        
        

    def build_my_network(self, num_elements_in_batch, shapes, 
                             output_dim, batch_size, inputs):
        """
        Build a network consistent with each type of inputs
        """
        if ("gpu" in theano.config.device):
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
            conv2DFunc = Conv2DCCLayer
        else:
            conv2DFunc = lasagne.layers.Conv2DLayer

        l_outs_conv=[]
        for i, element_shape in enumerate( zip(num_elements_in_batch, shapes) ):
            
            if(len(element_shape[1])>1): #frames
                # Building here for 3D
                l_in = lasagne.layers.InputLayer(
                    shape=(batch_size, element_shape[0], element_shape[1][0], element_shape[1][1]), 
                    input_var=inputs[i],
                )
                
                l_conv1 = conv2DFunc(
                    l_in,
                    num_filters=32,
                    filter_size=(1, 1),#filter_size=(8, 8),
                    stride=(1, 1),#stride=(4, 4),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(), # Defaults to Glorot
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                
                l_conv2 = conv2DFunc(
                    l_conv1,
                    num_filters=64,
                    filter_size=(1, 1),#filter_size=(4, 4),
                    stride=(1, 1),#stride=(2, 2),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                
                l_conv3 = conv2DFunc(
                    l_conv2,
                    num_filters=64,
                    filter_size=(3, 3),
                    stride=(1, 1),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                l_outs_conv.append(l_conv3)

            elif(element_shape[1][0]>1): # vectors
                # Building here for  2D
                l_in = lasagne.layers.InputLayer(
                    shape=(batch_size, 1, element_shape[0], element_shape[1][0]), 
                    input_var=inputs[i].reshape((batch_size, 1, element_shape[0], element_shape[1][0])),
                )
                
                l_conv1 = conv2DFunc(
                    l_in,
                    num_filters=32,
                    filter_size=(1, 1),#filter_size=(8, 8),
                    stride=(1, 1),#stride=(4, 4),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(), # Defaults to Glorot
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                
                l_conv2 = conv2DFunc(
                    l_conv1,
                    num_filters=64,
                    filter_size=(1, 1),#filter_size=(4, 4),
                    stride=(1, 1),#stride=(2, 2),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                
                l_conv3 = conv2DFunc(
                    l_conv2,
                    num_filters=64,
                    filter_size=(1, 1),
                    stride=(1, 1),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                    dimshuffle=True
                )
                l_outs_conv.append(l_conv3)

            elif(element_shape[1][0]==1 and element_shape[0]>3):
                # Building here for  1D
                l_in = lasagne.layers.InputLayer(
                    shape=(batch_size, 1, element_shape[0]), 
                    input_var=inputs[i].reshape((batch_size, 1, element_shape[0])),
                )
                
                l_conv1 = lasagne.layers.Conv1DLayer(
                    l_in,
                    num_filters=8,#32,
                    filter_size=2,#filter_size=(8, 8),
                    stride=1,#stride=(4, 4),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(), # Defaults to Glorot
                    b=lasagne.init.Constant(.0),
                    #dimshuffle=True
                )
                
                l_conv2 = lasagne.layers.Conv1DLayer(
                    l_conv1,
                    num_filters=8,#64,
                    filter_size=2,#filter_size=(4, 4),
                    stride=1,#stride=(2, 2),
                    nonlinearity=lasagne.nonlinearities.rectify,
                    W=lasagne.init.HeUniform(),
                    b=lasagne.init.Constant(.0),
                    #dimshuffle=True
                )
                
                l_outs_conv.append(l_conv2)
            
            elif(element_shape[1][0]==1 and element_shape[0]<=3):
                # Building here for 1D simple
                l_in = lasagne.layers.InputLayer(
                    shape=(batch_size, 1, element_shape[0]), 
                    input_var=inputs[i].reshape((batch_size, 1, element_shape[0])),
                )
                                
                l_outs_conv.append(l_in)
        
        ## Custom merge of layers
        ## NB : l_output_conv=lasagne.layers.MergeLayer(l_outs_conv) gives NOT IMPLEMENTED ERROR
        output_conv=lasagne.layers.get_output(l_outs_conv[0]).flatten().reshape(( batch_size,np.prod(l_outs_conv[0].output_shape[1:]) ))       
        shapes=[np.prod(l_outs_conv[0].output_shape[1:])]
        
        if (len(l_outs_conv)>1):
            for l_out_conv in l_outs_conv[1:]:
                output_conv=T.concatenate(( output_conv , lasagne.layers.get_output(l_out_conv).flatten().reshape(( batch_size,np.prod(l_out_conv.output_shape[1:]) )) ) , axis=1)
                shapes.append(np.prod( l_out_conv.output_shape[1:] ))
        
        shape=sum(shapes)
        
        l_output_conv = lasagne.layers.InputLayer(
            shape=( [batch_size, shape] ), 
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
            num_units=output_dim,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.0)
        )

        return l_out,l_outs_conv,shapes


if __name__ == '__main__':
    pass
