"""
Code for general deep Q-learning that can take as inputs scalars, vectors and matrices

Authors: Vincent Francois-Lavet, David Taralla

Inspired from "Human-level control through deep reinforcement learning",
Nature, 518(7540):529-533, February 2015
"""

import numpy as np
import theano
import theano.tensor as T
from .updates import deepmind_rmsprop
from ..base_classes import QNetwork

from .theano_layers import ConvolutionalLayer,HiddenLayer

    
class MyQNetwork(QNetwork):
    """
    Deep Q-learning network using Theano
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
        
        q_vals, self.params, shape_after_conv = self._build(network_type, states)
        
        print("Number of neurons after spatial and temporal convolution layers: {}".format(shape_after_conv))

        next_q_vals, self.next_params, shape_after_conv = self._build(network_type, next_states)
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
        
        
        max_next_q_vals=T.max(next_q_vals, axis=1, keepdims=True)
        
        T_ones_like=T.ones_like(T.ones_like(terminals) - terminals)
                
        target = rewards + T_ones_like * thediscount * max_next_q_vals

        q_val=q_vals[T.arange(batchSize), actions.reshape((-1,))].reshape((-1, 1))
        # Note : Strangely (target - q_val) lead to problems with python 3.5, theano 0.8.0rc and floatX=float32...
        diff = - q_val + target 

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
        # FIXME

        return None,None
    
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
        for i,(param,next_param) in enumerate(zip(self.params, self.next_params)):
            next_param.set_value(param.get_value())        
        

    def _buildG_DQN_0(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        outs_conv_shapes=[]
        
        for i, dim in enumerate(self._inputDimensions):
            nfilter=[]
            
            # - observation[i] is a FRAME -
            if len(dim) == 3: 

                ### First layer
                newR = dim[1]
                newC = dim[2]
                fR=8  # filter Rows
                fC=8  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(32)
                stride_size=4
                l_conv1 = ConvolutionalLayer(
                    rng=self._randomState,
                    input=inputs[i].reshape((self._batchSize,dim[0],newR,newC)),
                    filter_shape=(nfilter[0],dim[0],fR,fC),
                    image_shape=(self._batchSize,dim[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv1)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                ### Second layer
                fR=4  # filter Rows
                fC=4  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(64)
                stride_size=2
                l_conv2 = ConvolutionalLayer(
                    rng=self._randomState,
                    input=l_conv1.output.reshape((self._batchSize,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batchSize,nfilter[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv2)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                ### Third layer
                fR=3  # filter Rows
                fC=3  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(64)
                stride_size=1
                l_conv3 = ConvolutionalLayer(
                    rng=self._randomState,
                    input=l_conv2.output.reshape((self._batchSize,nfilter[1],newR,newC)),
                    filter_shape=(nfilter[2],nfilter[1],fR,fC),
                    image_shape=(self._batchSize,nfilter[1],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv3)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                outs_conv.append(l_conv3.output)
                outs_conv_shapes.append((nfilter[2],newR,newC))

                
            # - observation[i] is a VECTOR -
            elif len(dim) == 2 and dim[0] > 3:                
                
                newR = dim[0]
                newC = dim[1]
                
                fR=2  # filter Rows
                fC=1  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(16)
                stride_size=1

                l_conv1 = ConvolutionalLayer(
                    rng=self._randomState,
                    input=inputs[i].reshape((self._batchSize,1,newR,newC)),
                    filter_shape=(nfilter[0],1,fR,fC),
                    image_shape=(self._batchSize,1,newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)                    
                )                
                layers.append(l_conv1)
                
                newR = (newR - fR + 1 - pR) // stride_size + 1  # stride 2
                newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                fR=2  # filter Rows
                fC=2  # filter Column
                pR=1  # pool Rows
                pC=1  # pool Column
                nfilter.append(16)
                stride_size=1

                l_conv2 = ConvolutionalLayer(
                    rng=self._randomState,
                    input=l_conv1.output.reshape((self._batchSize,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batchSize,nfilter[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )                
                layers.append(l_conv2)
                
                newR = (newR - fR + 1 - pR) // stride_size + 1  # stride 2
                newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                outs_conv.append(l_conv2.output)
                outs_conv_shapes.append((nfilter[1],newR,newC))


            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    newR = 1
                    newC = dim[0]
                    
                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1

                    l_conv1 = ConvolutionalLayer(
                        rng=self._randomState,
                        input=inputs[i].reshape((self._batchSize,1,newR,newC)),
                        filter_shape=(nfilter[0],1,fR,fC),
                        image_shape=(self._batchSize,1,newR,newC),
                        poolsize=(pR,pC),
                        stride=(stride_size,stride_size)
                    )                
                    layers.append(l_conv1)
                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                    fR=1  # filter Rows
                    fC=2  # filter Column
                    pR=1  # pool Rows
                    pC=1  # pool Column
                    nfilter.append(8)
                    stride_size=1
                    
                    l_conv2 = ConvolutionalLayer(
                        rng=self._randomState,
                        input=l_conv1.output.reshape((self._batchSize,nfilter[0],newR,newC)),
                        filter_shape=(nfilter[1],nfilter[0],fR,fC),
                        image_shape=(self._batchSize,nfilter[0],newR,newC),
                        poolsize=(pR,pC),
                        stride=(stride_size,stride_size)
                    )                
                    layers.append(l_conv2)
                    
                    newC = (newC - fC + 1 - pC) // stride_size + 1  # stride 2

                    outs_conv.append(l_conv2.output)
                    outs_conv_shapes.append((nfilter[1],newC))
                    
                else:
                    if(len(dim) == 2):
                        outs_conv_shapes.append((dim[0],dim[1]))
                    elif(len(dim) == 1):
                        outs_conv_shapes.append((1,dim[0]))
                    outs_conv.append(inputs[i])
        
        
        ## Custom merge of layers
        output_conv = outs_conv[0].flatten().reshape((self._batchSize, np.prod(outs_conv_shapes[0])))
        shapes=np.prod(outs_conv_shapes[0])

        if (len(outs_conv)>1):
            for out_conv,out_conv_shape in zip(outs_conv[1:],outs_conv_shapes[1:]):
                output_conv=T.concatenate((output_conv, out_conv.flatten().reshape((self._batchSize, np.prod(out_conv_shape)))) , axis=1)
                shapes+=np.prod(out_conv_shape)
                shapes

                
        self.hiddenLayer1 = HiddenLayer(rng=self._randomState, input=output_conv,
                                       n_in=shapes, n_out=50,
                                       activation=T.tanh)                                       
        layers.append(self.hiddenLayer1)

        self.hiddenLayer2 = HiddenLayer(rng=self._randomState, input=self.hiddenLayer1.output,
                                       n_in=50, n_out=20,
                                       activation=T.tanh)
        layers.append(self.hiddenLayer2)

        self.outLayer = HiddenLayer(rng=self._randomState, input=self.hiddenLayer2.output,
                                       n_in=20, n_out=self._nActions,
                                       activation=None)
        layers.append(self.outLayer)

        # Grab all the parameters together.
        params = [param
                       for layer in layers
                       for param in layer.params]
        
        return self.outLayer.output, params, outs_conv_shapes

if __name__ == '__main__':
    pass
