"""
Neural network using Keras (called by q_net_keras)

.. Authors: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Convolution2D, MaxPooling2D, Reshape
import theano.tensor as T

class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    inputDimensions :
    n_Actions :
    randomState : numpy random number generator
    """
    def __init__(self, batchSize, inputDimensions, n_Actions, randomState):
        self._inputDimensions=inputDimensions
        self._batchSize=batchSize
        self._randomState=randomState
        self._nActions=n_Actions

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        inputs=[]
        #print "inputs"
        #print inputs

        for i, dim in enumerate(self._inputDimensions):
            nfilter=[]
            print i
            # - observation[i] is a FRAME - FIXME
            if len(dim) == 3: 
                model = Sequential()
                layers.append(model.layers[-1])
                model.add(Flatten())
                
            # - observation[i] is a VECTOR - FIXME
            elif len(dim) == 2 and dim[0] > 3:                                
                model = Sequential()
                layers.append(model.layers[-1])
                model.add(Flatten())
            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    print "here"
                    print dim[0]
                    # this returns a tensor
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    reshaped=Reshape((1,1,dim[0]), input_shape=(dim[0],))(input)
                    x = Convolution2D(8, 1, 2, border_mode='valid')(reshaped)
                    x = Convolution2D(8, 1, 2)(x)
                    
                    out = Flatten()(x)
                                        
                else:
                    print "here2"
                    print (dim[0])
                    
                    if(len(dim) == 2):
                    # this returns a tensor

                        input = Input(shape=(dim[0],dim[1],))
                        inputs.append(input)
                        out = Flatten()(input)

                    if(len(dim) == 1):
                        input = Input(shape=(dim[0],))
                        inputs.append(input)
                        out=input
                    
#                    layers.append(model.layers[-1])
#                    model.add(Flatten())
                #models_conv.append(model)

#            models_conv.append(model)
            outs_conv.append(out)
        print "inputs"
        print inputs
        print "outs_conv"
        print outs_conv
        ## Merge of layers

        x = merge(outs_conv, mode='concat')        
        
        # we stack a deep fully-connected network on top
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        out = Dense(self._nActions)(x)

        model = Model(input=inputs, output=out)
        layers=model.layers
        print layers
        
        
        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]
        print params
#        params.append([ param
#                    for layer in layers 
#                    for param in layer.non_trainable_weights])

        return model, params#.layers[-1].output #, params, None

if __name__ == '__main__':
    pass

