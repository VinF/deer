"""
CRAR Neural network using Keras

"""

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Permute, Add, Subtract, Dot, Multiply, Average, Lambda, Concatenate, BatchNormalization, merge, RepeatVector, AveragePooling2D
from keras import regularizers
#np.random.seed(111111)

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
    high_int_dim : Boolean
        Whether the abstract state should be high dimensional in the form of frames/vectors or whether it should 
        be low-dimensional
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, **kwargs):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._high_int_dim=kwargs["high_int_dim"]
        if(self._high_int_dim==True):
            self.n_channels_internal_dim=kwargs["internal_dim"] #dim[-3]
        else:
            self.internal_dim=kwargs["internal_dim"]    #2 for laby
                                                        #3 for catcher

    def encoder_model(self):
        """ Instantiate a Keras model for the encoder of the CRAR learning algorithm.
        
        The model takes the following as input 
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        
        Parameters
        -----------
        
    
        Returns
        -------
        Keras model with output x (= encoding of s)
    
        """
        outs_conv=[]
        inputs=[]

        for i, dim in enumerate(self._input_dimensions):
            # - observation[i] is a FRAME
            if len(dim) == 3 or len(dim) == 4:
                if(len(dim) == 4):
                    input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                    input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
                    x=Permute((2,3,1), input_shape=(dim[-4]*dim[-3],dim[-2],dim[-1]))(input)    #data_format='channels_last'
                else:
                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                    x=Permute((2,3,1), input_shape=(dim[-3],dim[-2],dim[-1]))(input)    #data_format='channels_last'

                if(dim[-2]>12 and dim[-1]>12):
                    self._pooling_encoder=6
                    x = Conv2D(8, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
                else:
                    self._pooling_encoder=1
                    
                if(self._high_int_dim==True):
                    x = Conv2D(self.n_channels_internal_dim, (1, 1), padding='same')(x)
                    out = x
                else:
                    out = Flatten()(x)
                
            # - observation[i] is a VECTOR
            elif len(dim) == 2:
                if dim[-3] > 3:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
                    reshaped=Reshape((dim[-3],dim[-2],1), input_shape=(dim[-3],dim[-2]))(input)     #data_format='channels_last'
                    x = Conv2D(16, (2, 1), activation='relu', border_mode='valid')(reshaped)    #Conv on the history
                    x = Conv2D(16, (2, 2), activation='relu', border_mode='valid')(x)           #Conv on the history & features
            
                    if(self._high_int_dim==True):
                        out = x
                    else:
                        out = Flatten()(x)
                else:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
                    out = Flatten()(input)
            
            # - observation[i] is a SCALAR -
            else:
                if dim[-3] > 3:
                    # this returns a tensor
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)
                    reshaped=Reshape((1,dim[-3],1), input_shape=(dim[-3],))(input)            #data_format='channels_last'
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(reshaped)  #Conv on the history
                    x = Conv2D(8, (1,2), activation='relu', border_mode='valid')(x)         #Conv on the history
                    
                    if(self._high_int_dim==True):
                        out = x
                    else:
                        out = Flatten()(x)
                                        
                else:
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)
                    out=input
                    
            outs_conv.append(out)
        
        if(self._high_int_dim==True):
            model = Model(inputs=inputs, outputs=outs_conv)

        if(self._high_int_dim==False):
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
        """ Instantiate a Keras model that provides the difference between two encoded pseudo-states
        
        The model takes the two following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        s2 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder
    
        Returns
        -------
        model with output the difference between the encoding of s1 and the encoding of s2
    
        """
        inputs=[]
        
        for j in range(2):
            for i, dim in enumerate(self._input_dimensions):
                if(len(dim) == 4):
                    input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                    input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
                elif(len(dim) == 3):
                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                elif len(dim) == 2:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
                else:
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)
        
        half = len(inputs)//2
        x1 = encoder_model(inputs[:half])
        x2 = encoder_model(inputs[half:])
        
        if (self._high_int_dim==True):
            x1=Flatten()(x1)
            x2=Flatten()(x2)
        x = Subtract()([x1,x2])
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def transition_model(self):
        """  Instantiate a Keras model for the transition between two encoded pseudo-states.
    
        The model takes as inputs:
        x : internal state
        a : int
            the action considered
        
        Parameters
        -----------
    
        Returns
        -------
        model that outputs the transition of (x,a)
    
        """
        if(self._high_int_dim==True):
            dim=self._input_dimensions[0] #FIXME
            inputs = [ Input(shape=((dim[-2] // self._pooling_encoder),(dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)), Input( shape=(self._n_actions,) ) ]     # data_format='channels_last'
            
            layers_action=inputs[1]
            layers_action=RepeatVector((dim[-2] // self._pooling_encoder)*(dim[-1] // self._pooling_encoder))(layers_action)
            layers_action=Reshape(((dim[-2] // self._pooling_encoder),(dim[-1] // self._pooling_encoder),self._n_actions))(layers_action)
            
            x = Concatenate(axis=-1)([layers_action,inputs[0]])
            
            x = Conv2D(16, (1, 1), padding='same', activation='tanh')(x)
            x = Conv2D(32, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(64, (3, 3), padding='same', activation='tanh')(x)
            x = Conv2D(32, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(16, (1, 1), padding='same', activation='tanh')(x)
            x = Conv2D(self.n_channels_internal_dim, (1, 1), padding='same')(x)
            x = Add()([inputs[0],x])
        else:
            inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ]     # x

            x = Concatenate()(inputs)
            x = Dense(10, activation='tanh')(x)
            x = Dense(30, activation='tanh')(x)
            x = Dense(30, activation='tanh')(x)
            x = Dense(10, activation='tanh')(x)
            x = Dense(self.internal_dim)(x)
            x = Add()([inputs[0],x])
        
        model = Model(inputs=inputs, outputs=x)
        
        return model

    def diff_Tx_x_(self,encoder_model,transition_model,plan_depth=0):
        """ For plan_depth=0, instantiate a Keras model that provides the difference between T(E(s1),a) and E(s2).
        Note that it gives 0 if the transition leading to s2 is terminal (we don't need to fit the transition if 
        it is terminal).
        
        For plan_depth=0, the model takes the four following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        s2 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s1
        terminal : boolean
            Whether the transition leading to s2 is terminal
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        transition_model: instantiation of a Keras model for the transition (T)
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2 
        (input a is then a list of actions)
    
        Returns
        -------
        model with output Tx (= model estimate of x')
    
        """
        inputs=[]
        for j in range(2):
            for i, dim in enumerate(self._input_dimensions):
                if(len(dim) == 4):
                    input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                    input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
                elif(len(dim) == 3):
                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
                elif len(dim) == 2:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
                else:
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)

        half = len(inputs)//2
        enc_x = encoder_model(inputs[:half]) #s --> x
        enc_x_ = encoder_model(inputs[half:]) #s --> x

        Tx= enc_x
        for d in range(plan_depth+1):
            inputs.append(Input(shape=(self._n_actions,)))
            Tx= transition_model([Tx,inputs[-1]])
                        
        x = Subtract()([Tx,enc_x_])

        input = Input(shape=(1,)) # 1-terminals (0 if transition is terminal)
        inputs.append(input)
        x = Multiply()([x,inputs[-1]])# set to 0 if terminal because we don't care about fitting that transition
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def force_features(self,encoder_model,transition_model,plan_depth=0):
        """ Instantiate a Keras model that provides the vector of the transition at E(s1). It is calculated as the different between E(s1) and E(T(s1)). 
        Used to force the directions of the transitions.
        
        The model takes the four following inputs:
        s1 : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s1
        
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        transition_model: instantiation of a Keras model for the transition (T)
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions between s1 and s2 
        (input a is then a list of actions)
            
        Returns
        -------
        model with output E(s1)-T(E(s1))
    
        """
        inputs=[]
        for i, dim in enumerate(self._input_dimensions):
            if(len(dim) == 4):
                input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
                input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
            elif(len(dim) == 3):
                input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
            elif len(dim) == 2:
                input = Input(shape=(dim[-3],dim[-2]))
                inputs.append(input)
            else:
                input = Input(shape=(dim[-3],))
                inputs.append(input)

        enc_x = encoder_model(inputs[:]) #s --> x
        
        Tx= enc_x
        for d in range(plan_depth+1):
            inputs.append(Input(shape=(self._n_actions,)))
            Tx= transition_model([Tx,inputs[-1]])
        
        diff_features = Subtract()([Tx,enc_x]) # Modification of the features after (sequence of) action(s)
        
        model = Model(inputs=inputs, outputs=diff_features )
        
        return model

    def float_model(self):
        """ Instantiate a Keras model for fitting a float from x.
                
        The model takes the following inputs:
        x : internal state
        a : int
            the action considered at x
        
        Parameters
        -----------
            
        Returns
        -------
        model that outputs a float
    
        """
        
        if(self._high_int_dim==True):
            dim=self._input_dimensions[0] #FIXME
            inputs = [ Input(shape=((dim[-2] // self._pooling_encoder),(dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)), Input( shape=(self._n_actions,) ) ]     #data_format='channels_last'
            
            layers_action=inputs[1]
            layers_action=RepeatVector((dim[-2] // self._pooling_encoder)*(dim[-1] // self._pooling_encoder))(layers_action)
            layers_action=Reshape(((dim[-2] // self._pooling_encoder),(dim[-1] // self._pooling_encoder),self._n_actions))(layers_action)

            
            x = Concatenate(axis=-1)([layers_action,inputs[0]])
            x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
            x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(4, (1, 1), padding='same', activation='tanh')(x)

            # we stack a deep fully-connected network on top
            x = Flatten()(x)
            x = Dense(200, activation='tanh')(x)
        else:
            inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ] #x
            x = Concatenate()(inputs)#,axis=-1)
            x = Dense(10, activation='tanh')(x)
       
        x = Dense(50, activation='tanh')(x)
        x = Dense(20, activation='tanh')(x)
        
        out = Dense(1)(x)
                
        model = Model(inputs=inputs, outputs=out)
        
        return model

    def full_float_model(self,encoder_model,float_model,plan_depth=0,transition_model=None):
        """ Instantiate a Keras model for fitting a float from s.
                
        The model takes the four following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length (plan_depth+1)
            the action(s) considered at s
                
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        float_model: instantiation of a Keras model for fitting a float from x
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
            
        Returns
        -------
        model with output the reward r
        """
        
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if(len(dim) == 4):
                input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
                input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
            elif(len(dim) == 3):
                input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
            elif len(dim) == 2:
                input = Input(shape=(dim[-3],dim[-2]))
                inputs.append(input)
            else:
                input = Input(shape=(dim[-3],))
                inputs.append(input)
        
        enc_x = encoder_model(inputs[:]) #s --> x
        
        Tx= enc_x
        for d in range(plan_depth):
            inputs.append(Input(shape=(self._n_actions,)))
            Tx= transition_model([Tx,inputs[-1]])

        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        
        out = float_model([Tx]+inputs[-1:])

        model = Model(inputs=inputs, outputs=out)
        
        return model

    def Q_model(self):
        """ Instantiate a  a Keras model for the Q-network from x.

        The model takes the following inputs:
        x : internal state

        Parameters
        -----------
            
        Returns
        -------
        model that outputs the Q-values for each action
        """
        if(self._high_int_dim==True):
            inputs=[]
            outs_conv=[]
            for i, dim in enumerate(self._input_dimensions):
                # - observation[i] is a FRAME
                if len(dim) == 3 or len(dim) == 4:
                    input = Input(shape=((dim[-2] // self._pooling_encoder),(dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)) #data_format is already 'channels_last'
                    inputs.append(input)
                    x = input     #data_format is already 'channels_last'
            
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(4, (1, 1), padding='same', activation='tanh')(x)
                    out = (x)
                else:
                    print ("FIXME")
                        
                outs_conv.append(out)
            
            if len(outs_conv)>1:
                x = merge(outs_conv, mode='concat')
            else:
                x= outs_conv [0]
            
            # we stack a deep fully-connected network on top
            x = Flatten()(x)
            x = Dense(200, activation='tanh')(x)
        else:
            inputs = [ Input( shape=(self.internal_dim,) ) ] #x
            x = Dense(20, activation='tanh')(inputs[0])
        
        # we stack a deep fully-connected network on top
        x = Dense(50, activation='tanh')(x)
        x = Dense(20, activation='tanh')(x)
        
        out = Dense(self._n_actions)(x)
                
        model = Model(inputs=inputs, outputs=out)
        
        return model


    def full_Q_model(self, encoder_model, Q_model, plan_depth=0, transition_model=None, R_model=None, discount_model=None):
        """ Instantiate a  a Keras model for the Q-network from s.

        The model takes the following inputs:
        s : list of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        a : list of ints with length plan_depth; if plan_depth=0, there isn't any input for a.
            the action(s) considered at s
    
        Parameters
        -----------
        encoder_model: instantiation of a Keras model for the encoder (E)
        Q_model: instantiation of a Keras model for the Q-network from x.
        plan_depth: if>1, it provides the possibility to consider a sequence of transitions following s 
        (input a is then a list of actions)
        transition_model: instantiation of a Keras model for the transition (T)
        R_model: instantiation of a Keras model for the reward
        discount_model: instantiation of a Keras model for the discount
            
        Returns
        -------
        model with output the Q-values
        """
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if(len(dim) == 4):
                input = Input(shape=(dim[-4],dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
                input = Reshape((dim[-4]*dim[-3],dim[-2],dim[-1]), input_shape=(dim[-4],dim[-3],dim[-2],dim[-1]))(input)
            elif(len(dim) == 3):
                input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
            elif len(dim) == 2:
                input = Input(shape=(dim[-3],dim[-2]))
                inputs.append(input)
            else:
                input = Input(shape=(dim[-3],))
                inputs.append(input)
        
        out = encoder_model(inputs)

        disc_plan = None
        disc_rewards=[]
        for d in range(plan_depth):
            inputs.append(Input(shape=(self._n_actions,)))
            reward=R_model([out]+inputs[-1:])
            if(disc_plan == None):
                disc_rewards.append(reward)
            else:
                disc_rewards.append(Multiply()([disc_plan,reward]))
            discount=discount_model([out]+inputs[-1:])
            if(disc_plan == None):
                disc_plan=discount
            else:
                disc_plan=Multiply()([disc_plan,discount]) #disc_model([out]+inputs[-1:])

            out=transition_model([out]+inputs[-1:])
        
        if(plan_depth==0):
            Q_estim=Q_model(out)
        else:
            Q_estim = Multiply()([disc_plan,Q_model(out)])
            Q_estim = Add()([Q_estim]+disc_rewards)

        model = Model(inputs=inputs, outputs=Q_estim)
        
        return model

if __name__ == '__main__':
    pass
    