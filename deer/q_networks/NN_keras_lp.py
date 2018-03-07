"""
Neural network using Keras (called by q_net_keras)
.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, Reshape, Permute, Add, Subtract, Dot, Multiply, Average, Lambda, Concatenate, BatchNormalization, merge
from keras import regularizers
np.random.seed(102912)

class NN():
    """
    Deep Q-learning network using Keras
    
    Parameters
    -----------
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent
    input_dimensions :
    n_actions :
    random_state : numpy random number generator
    action_as_input : Boolean
        Whether the action is given as input or as output
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, action_as_input=False):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._action_as_input=action_as_input
        self.internal_dim=3 # size random vector
        self._rand_vect_size=5 # size output distribution

    def encoder_model(self):
        """
    
        Parameters
        -----------
        s
    
        Returns
        -------
        model with output x (= encoding of s)
    
        """
        layers=[]
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)
                x=Permute((2,3,1), input_shape=(dim[0],dim[1],dim[2]))(input)    #data_format='channels_last'
                x = Conv2D(4, (2, 2), padding='same', activation='tanh')(x)
                #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                x = Conv2D(8, (3, 3), padding='same', activation='tanh')(x)
                x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                #x = Conv2D(16, (4, 4), padding='same', activation='tanh')(x)
                #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                
                out = Flatten()(x)
                
            # - observation[i] is a VECTOR
            elif len(dim) == 2:
                if dim[0] > 3:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    reshaped=Reshape((dim[0],dim[1],1), input_shape=(dim[0],dim[1]))(input) 
                    x = Conv2D(16, (2, 1), activation='relu', border_mode='valid')(reshaped)#Conv on the history
                    x = Conv2D(16, (2, 2), activation='relu', border_mode='valid')(x)       #Conv on the history & features

                    out = Flatten()(x)
                else:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
                    out = Flatten()(input)

            # - observation[i] is a SCALAR -
            else:
                if dim[0] > 3:
                    # this returns a tensor
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    reshaped=Reshape((1,dim[0],1), input_shape=(dim[0],))(input)  
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(reshaped)  #Conv on the history
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(x)         #Conv on the history
                    
                    out = Flatten()(x)
                                        
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
                    out=input
                    
            outs_conv.append(out)

        if (self._action_as_input==True):
            if ( isinstance(self._n_actions,int)):
                print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
            else:
                input = Input(shape=(len(self._n_actions),))
                inputs.append(input)
                outs_conv.append(input)
        
        if len(outs_conv)>1:
            x = merge(outs_conv, mode='concat')
        else:
            x= outs_conv [0]
        
        # we stack a deep fully-connected network on top
        x = Dense(200, activation='tanh')(x)
        x = Dense(100, activation='tanh')(x)
        x = Dense(50, activation='tanh')(x)
        x = Dense(10, activation='tanh')(x)
        
        x = Dense(self.internal_dim)(x)#, activity_regularizer=regularizers.l2(0.00001))(x) #, activation='relu'
        
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def encoder_diff_model(self,encoder_model):
        """
    
        Parameters
        -----------
        s
    
        Returns
        -------
        model with output x (= encoding of s)
    
        """
        inputs=[]
        
        for j in range(2):
            for i, dim in enumerate(self._input_dimensions):
                if len(dim) == 3:
                    input = Input(shape=(dim[0],dim[1],dim[2]))
                    inputs.append(input)
            
                elif len(dim) == 2:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
            
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
        
        half = len(inputs)/2
        x1 = encoder_model(inputs[:half])
        x2 = encoder_model(inputs[half:])
        
        x = Subtract()([x1,x2])
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def transition_model(self):
        """
    
        Parameters
        -----------
        x
        a
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ] #x

        x = Concatenate()(inputs)#,axis=-1)
        x = Dense(10, activation='tanh')(x) #5,15
        x = Dense(30, activation='tanh')(x) # ,30
        x = Dense(30, activation='tanh')(x) # ,30
        x = Dense(10, activation='tanh')(x) # ,30
        #x = Dense(5, activation='tanh')(x) #5,15
        x = Dense(self.internal_dim)(x)#, activity_regularizer=regularizers.l2(0.00001))(x) #, activation='relu'
        x = Add()([inputs[0],x])
        
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def transition_model2(self):
        """
    
        Parameters
        -----------
        x
        a
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ] #x

        x = Concatenate()(inputs)#,axis=-1)
        x = Dense(10, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dense(50, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dense(10, activation='tanh')(x)
        x = BatchNormalization()(x)
        x = Dense(self.internal_dim)(x)#, activity_regularizer=regularizers.l2(0.00001))(x) #, activation='relu'
        x = Add()([inputs[0],x])
        
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def diff_Tx_x_(self,encoder_model,transition_model):
        """
    
        Parameters
        -----------
        s
        a
        s'
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs=[]
        for j in range(2):
            for i, dim in enumerate(self._input_dimensions):
                if len(dim) == 3:
                    input = Input(shape=(dim[0],dim[1],dim[2]))
                    inputs.append(input)
            
                elif len(dim) == 2:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
            
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)

        half = len(inputs)/2
        enc_x = encoder_model(inputs[:half]) #s --> x
        enc_x_ = encoder_model(inputs[half:]) #s --> x

        input = Input(shape=(self._n_actions,))
        inputs.append(input)
                
        Tx= transition_model([enc_x,inputs[-1]])
        
        x = Subtract()([Tx,enc_x_])
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def diff_s_s_(self,encoder_model):
        """
    
        Parameters
        -----------
        s
        a
        random z
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs=[]
        
        for j in range(2):
            for i, dim in enumerate(self._input_dimensions):
                if len(dim) == 3:
                    input = Input(shape=(dim[0],dim[1],dim[2]))
                    inputs.append(input)
            
                elif len(dim) == 2:
                    input = Input(shape=(dim[0],dim[1]))
                    inputs.append(input)
            
                else:
                    input = Input(shape=(dim[0],))
                    inputs.append(input)
        
        half = len(inputs)/2
        enc_x = encoder_model(inputs[:half]) #s --> x #FIXME
        enc_x_ = encoder_model(inputs[half:]) #s --> x
        
        x = Subtract()([enc_x,enc_x_])
        x = Dot(axes=-1, normalize=False)([x,x])
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def diff_sa_sa(self,encoder_model,transition_model):
        """
    
        Parameters
        -----------
        s
        a
        rand_a
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)

            elif len(dim) == 2:
                input = Input(shape=(dim[0],dim[1]))
                inputs.append(input)

            else:
                input = Input(shape=(dim[0],))
                inputs.append(input)
        
        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        
        enc_x = encoder_model(inputs[:-2]) #s --> x
        Tx= transition_model([enc_x,inputs[-2]])
        rand_Tx= transition_model([enc_x,inputs[-1]])
                
        x = Subtract()([Tx,rand_Tx])
        x = Dot(axes=-1, normalize=False)([x,x])
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def diff_Tx(self,transition_model):
        """
    
        Parameters
        -----------
        x
        a
        x
        a
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ), Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) )] #x,a,x,a
        
        #identity_mat=inputs[2]#K.constant(np.diag(np.ones(self._n_actions)), name="identity_mat")
        Tx = transition_model(inputs[:2])
        Tx2 = transition_model(inputs[2:])
        
        #tile_x=K.tile(inputs[0],(self._n_actions,1))        
        #Tx_ = transition_model([tile_x]+[identity_mat])
        
        x = Subtract()([Tx,Tx2])
        x = Dot(axes=-1, normalize=False)([x,x])
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def R_model(self):
        """
        Build a network consistent with each type of inputs

        Parameters
        -----------
        x
        a
    
        Returns
        -------
        r
        """
        
        inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ] #x
        
        x = Concatenate()(inputs)#,axis=-1)
        x = Dense(10, activation='tanh')(x)
        x = Dense(20, activation='tanh')(x)
        x = Dense(10, activation='tanh')(x)
        
        out = Dense(1)(x)
                
        model = Model(inputs=inputs, outputs=out)
        
        return model

    def full_R_model(self,encoder_model,R_model):
        """
        Maps internal state to immediate rewards

        Parameters
        -----------
        s
        a
        (noise in abstract state space) : FIXME
    
        Returns
        -------
        r
        """
        
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)

            elif len(dim) == 2:
                input = Input(shape=(dim[0],dim[1]))
                inputs.append(input)

            else:
                input = Input(shape=(dim[0],))
                inputs.append(input)
        
        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        
        enc_x = encoder_model(inputs[:-1]) #s --> x
                
        out = R_model([enc_x]+inputs[-1:])
                
        model = Model(inputs=inputs, outputs=out)
        
        return model

    def Q_model(self):
        
        inputs = [ Input( shape=(self.internal_dim,) ) ] #x
        
        #if (self._action_as_input==True):
        #    if ( isinstance(self._n_actions,int)):
        #        print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
        #    else:
        #        input = Input(shape=(len(self._n_actions),))
        #        inputs.append(input)
                
        #x = Add()([x,inputs[-1]]) #????
        
        # we stack a deep fully-connected network on top
        x = Dense(20, activation='tanh')(inputs[0])
        x = Dense(50, activation='tanh')(x)
        x = Dense(20, activation='tanh')(x)
        
        #if (self._action_as_input==False):
        #    if ( isinstance(self._n_actions,int)):
        out = Dense(self._n_actions)(x)
        #    else:
        #        out = Dense(len(self._n_actions))(x)
        #else:
        #    out = Dense(1)(x)
                
        model = Model(inputs=inputs, outputs=out)
        
        return model


    def full_Q_model(self, encoder_model, Q_model):
        """
        Build a network consistent with each type of inputs

        Parameters
        -----------
        s
        noise in abstract state space
    
        Returns
        -------
        model with output Tx (= model estimate of x')
        """
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3:
                input = Input(shape=(dim[0],dim[1],dim[2]))
                inputs.append(input)

            elif len(dim) == 2:
                input = Input(shape=(dim[0],dim[1]))
                inputs.append(input)

            else:
                input = Input(shape=(dim[0],))
                inputs.append(input)
                
        out = encoder_model(inputs)
        input = Input(shape=(self.internal_dim,))
        inputs.append(input)
                
        x=Add()([out,inputs[-1]]) # adding noise in the abstract state space
        
        out = Q_model(out)

        model = Model(inputs=inputs, outputs=out)
        
        return model

if __name__ == '__main__':
    pass
    