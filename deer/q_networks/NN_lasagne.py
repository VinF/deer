"""
Neural network using Lasagne (called by q_net_theano)

.. Authors: Vincent Francois-Lavet, David Taralla
"""

import lasagne
import numpy as np
import theano
import theano.tensor as T

    
class NN():
    """
    Deep Q-learning network using Lasagne on top of Theano
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        
    def _buildDQN(self, inputs):
        """
        Build a network consistent with each type of inputs
        """
        if ("gpu" in theano.config.device):
            from lasagne.layers.cuda_convnet import Conv2DCCLayer
            conv2DFunc = Conv2DCCLayer
        else:
            conv2DFunc = lasagne.layers.Conv2DLayer

        l_outs_conv=[]
        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME -
            if len(dim) == 3: 
                # Building here for 3D
                l_in = lasagne.layers.InputLayer(
                    shape=(self._batch_size,) + dim,
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
                    shape=(self._batch_size, 1) + dim,
                    input_var=inputs[i].reshape((self._batch_size, 1) + dim),
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
                        shape=(self._batch_size, 1) + dim,
                        input_var=inputs[i].reshape((self._batch_size, 1) + dim),
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
                        shape=(self._batch_size, 1) + dim,
                        input_var=inputs[i].reshape((self._batch_size, 1) + dim),
                    )
                                
                    l_outs_conv.append(l_in)

        ## Custom merge of layers
        ## NB : l_output_conv=lasagne.layers.MergeLayer(l_outs_conv) gives NOT IMPLEMENTED ERROR
        output_conv = lasagne.layers.get_output(l_outs_conv[0]).flatten().reshape((self._batch_size, np.prod(l_outs_conv[0].output_shape[1:])))
        shapes = [np.prod(l_outs_conv[0].output_shape[1:])]

        if (len(l_outs_conv)>1):
            for l_out_conv in l_outs_conv[1:]:
                output_conv=T.concatenate((output_conv, lasagne.layers.get_output(l_out_conv).flatten().reshape((self._batch_size, np.prod(l_out_conv.output_shape[1:])))) , axis=1)
                shapes.append(np.prod(l_out_conv.output_shape[1:]))

        shape = sum(shapes)

        l_output_conv = lasagne.layers.InputLayer(
            shape=([self._batch_size, shape]),
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
            num_units=self._n_actions,
            nonlinearity=None,
            W=lasagne.init.HeUniform(),
            b=lasagne.init.Constant(.0)
        )

        params = lasagne.layers.helper.get_all_params(l_out)

        for conv_param in l_outs_conv:
            for p in lasagne.layers.helper.get_all_params(conv_param):
                params.append(p)

        
        return lasagne.layers.get_output(l_out), params, shapes


if __name__ == '__main__':
    pass
