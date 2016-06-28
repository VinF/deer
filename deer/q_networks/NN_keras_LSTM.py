"""
Neural network with LSTM's using Keras (called by q_net_keras)

.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, merge, Activation, Convolution2D, MaxPooling2D, Reshape
from keras.layers.recurrent import LSTM
import theano.tensor as T

class NN():
    """
    Deep Q-learning network with LSTM's using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions : tuples
    n_actions : int
    random_state : numpy random number generator
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions

    def _buildDQN(self):
        """
        Build a network consistent with each type of inputs
        """
        layers=[]
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)
                x = Convolution2D(32, 8, 8, border_mode='valid')(input)
                x = MaxPooling2D(pool_size=(4, 4), strides=None, border_mode='valid')(x)
                x = Convolution2D(64, 4, 4, border_mode='valid')(x)
                x = MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')(x)
                x = Convolution2D(64, 3, 3)(x)
                
                # We may add here LSTM's after having flatten the last two dimensions
                
                x = Flatten()(x)           

            # - observation[i] is a VECTOR
            if len(dim) == 2:
                input = Input(shape=(dim[0],dim[1]))
                inputs.append(input)
                
                if dim[0] > 3:

                    x = LSTM(16,
                        activation='relu',
                        return_sequences=True)(input)
                    x = LSTM(16,
                        activation='relu',
                        return_sequences=False)(x) # Structure many-to-one

                else:
                    x=input
                    x = Flatten()(x)

            # - observation[i] is a SCALAR -
            elif(len(dim) == 1):
            
                input = Input(shape=(dim[0],))
                inputs.append(input)
                input = Reshape((dim[0],1))(input)
                
                if dim[0] > 3:                    
                    x = LSTM(8,
                            activation='relu',
                            return_sequences=True)(input)
                    x = LSTM(8,
                            activation='relu',
                            return_sequences=False)(x) # Structure many-to-one
                else:
                    x=input
                    x = Flatten()(x)
                            
            outs_conv.append(x)
        
        if len(outs_conv)>1:
            x = merge(outs_conv, mode='concat')
        else:
            x= outs_conv [0]
                    
        x = Dense(50, activation='relu')(x)
        x = Dense(20, activation='relu')(x)
        out = Dense(self._n_actions)(x)

        model = Model(input=inputs, output=out)
        layers=model.layers

        # Grab all the parameters together.
        params = [ param
                    for layer in layers 
                    for param in layer.trainable_weights ]

        return model, params

if __name__ == '__main__':
    pass

