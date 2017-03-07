"""
Neural network using Theano (called by q_net_theano)

.. Authors: Vincent Francois-Lavet, David Taralla
"""

import numpy as np
import theano
import theano.tensor as T

from .theano_layers import ConvolutionalLayer,HiddenLayer

    
class NN():
    """
    Deep Q-learning network using Theano
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_Actions :
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
        layers=[]
        outs_conv=[]
        outs_conv_shapes=[]
        
        for i, dim in enumerate(self._input_dimensions):
            nfilter=[]
            
            # - observation[i] is a FRAME -
            if len(dim) == 3: 

                ### First layer
                newR = dim[1]
                newC = dim[2]
                fR=4  # filter Rows
                fC=4  # filter Column
                pR=2  # pool Rows
                pC=2  # pool Column
                nfilter.append(8)
                stride_size=2
                l_conv1 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=inputs[i].reshape((self._batch_size,dim[0],newR,newC)),
                    filter_shape=(nfilter[0],dim[0],fR,fC),
                    image_shape=(self._batch_size,dim[0],newR,newC),
                    poolsize=(pR,pC),
                    stride=(stride_size,stride_size)
                )
                layers.append(l_conv1)

                newR = (newR - fR + 1 - pR) // stride_size + 1
                newC = (newC - fC + 1 - pC) // stride_size + 1

                ### Second layer
                fR=4  # filter Rows
                fC=4  # filter Column
                pR=2  # pool Rows
                pC=2  # pool Column
                nfilter.append(16)
                stride_size=2
                l_conv2 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batch_size,nfilter[0],newR,newC),
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
                nfilter.append(16)
                stride_size=1
                l_conv3 = ConvolutionalLayer(
                    rng=self._random_state,
                    input=l_conv2.output.reshape((self._batch_size,nfilter[1],newR,newC)),
                    filter_shape=(nfilter[2],nfilter[1],fR,fC),
                    image_shape=(self._batch_size,nfilter[1],newR,newC),
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
                    rng=self._random_state,
                    input=inputs[i].reshape((self._batch_size,1,newR,newC)),
                    filter_shape=(nfilter[0],1,fR,fC),
                    image_shape=(self._batch_size,1,newR,newC),
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
                    rng=self._random_state,
                    input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                    filter_shape=(nfilter[1],nfilter[0],fR,fC),
                    image_shape=(self._batch_size,nfilter[0],newR,newC),
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
                        rng=self._random_state,
                        input=inputs[i].reshape((self._batch_size,1,newR,newC)),
                        filter_shape=(nfilter[0],1,fR,fC),
                        image_shape=(self._batch_size,1,newR,newC),
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
                        rng=self._random_state,
                        input=l_conv1.output.reshape((self._batch_size,nfilter[0],newR,newC)),
                        filter_shape=(nfilter[1],nfilter[0],fR,fC),
                        image_shape=(self._batch_size,nfilter[0],newR,newC),
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
        output_conv = outs_conv[0].flatten().reshape((self._batch_size, np.prod(outs_conv_shapes[0])))
        shapes=np.prod(outs_conv_shapes[0])

        if (len(outs_conv)>1):
            for out_conv,out_conv_shape in zip(outs_conv[1:],outs_conv_shapes[1:]):
                output_conv=T.concatenate((output_conv, out_conv.flatten().reshape((self._batch_size, np.prod(out_conv_shape)))) , axis=1)
                shapes+=np.prod(out_conv_shape)
                shapes

                
        self.hiddenLayer1 = HiddenLayer(rng=self._random_state, input=output_conv,
                                       n_in=shapes, n_out=50,
                                       activation=T.tanh)                                       
        layers.append(self.hiddenLayer1)

        self.hiddenLayer2 = HiddenLayer(rng=self._random_state, input=self.hiddenLayer1.output,
                                       n_in=50, n_out=20,
                                       activation=T.tanh)
        layers.append(self.hiddenLayer2)

        self.outLayer = HiddenLayer(rng=self._random_state, input=self.hiddenLayer2.output,
                                       n_in=20, n_out=self._n_actions,
                                       activation=None)
        layers.append(self.outLayer)

        # Grab all the parameters together.
        params = [param
                       for layer in layers
                       for param in layer.params]
        
        return self.outLayer.output, params, outs_conv_shapes

if __name__ == '__main__':
    pass
