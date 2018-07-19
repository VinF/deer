"""
Neural network using Keras (called by q_net_keras)
.. Author: Vincent Francois-Lavet
"""

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, Reshape, Permute, Add, Subtract, Dot, Multiply, Average, Lambda, Concatenate, BatchNormalization, merge, RepeatVector, AveragePooling2D
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
    high_int_dim : Boolean
        Whether the abstract state should be high dimensional in the form of frames/vectors or whether it should be low-dimensional
    """
    def __init__(self, batch_size, input_dimensions, n_actions, random_state, action_as_input=False, **kwargs):
        self._input_dimensions=input_dimensions
        self._batch_size=batch_size
        self._random_state=random_state
        self._n_actions=n_actions
        self._action_as_input=action_as_input
        self._high_int_dim=kwargs["high_int_dim"]
        if(self._high_int_dim==True):
            self.n_channels_internal_dim=kwargs["internal_dim"] #dim[-3]
        else:
            self.internal_dim=kwargs["internal_dim"]    #2 for laby
                                                        #3 for catcher

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
            print "dim enc"
            print dim
            if len(dim) == 3 or len(dim) == 4:
                input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                inputs.append(input)
                x=Permute((2,3,1), input_shape=(dim[-3],dim[-2],dim[-1]))(input)    #data_format='channels_last'
                if(dim[-2]>8 and dim[-1]>8):
                    self._pooling_encoder=6
                    #x = Conv2D(4, (3, 3), padding='same', activation='tanh')(x)
                    #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    #x = Conv2D(8, (1, 1), padding='same', activation='tanh')(x)
                    x = Conv2D(8, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(3, 3), strides=None, padding='same')(x)
                    #x = Conv2D(4, (2, 2), padding='same', activation='tanh')(x)
                    #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    #x = Conv2D(16, (4, 4), padding='same', activation='tanh')(x)
                    #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                else:
                    self._pooling_encoder=1
                    #x = Conv2D(8, (1, 1), padding='same', activation='tanh')(x)
                    #x = MaxPooling2D(pool_size=(self._pooling_encoder, self._pooling_encoder), strides=None, padding='same')(x)
                    
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

        if (self._high_int_dim==True):
            if ( isinstance(self._n_actions,int)):
                print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
            else:
                input = Input(shape=(len(self._n_actions),))
                inputs.append(input)
                outs_conv.append(input)
        
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
                if len(dim) == 3 or len(dim) == 4:
                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
            
                elif len(dim) == 2:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
            
                else:
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)
        
        half = len(inputs)/2
        x1 = encoder_model(inputs[:half])
        x2 = encoder_model(inputs[half:])
        
        if (self._high_int_dim==True):
            x1=Flatten()(x1)
            x2=Flatten()(x2)
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
        if(self._high_int_dim==True):
            dim=self._input_dimensions[0] #FIXME
            inputs = [ Input(shape=(-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)), Input( shape=(self._n_actions,) ) ]     # data_format='channels_last'
            print inputs[0]._keras_shape
            print inputs[1]._keras_shape
            
            layers_action=inputs[1]
            layers_action=RepeatVector(-(-dim[-2] // self._pooling_encoder)*-(-dim[-1] // self._pooling_encoder))(layers_action)#K.repeat_elements(layers_action,rep=dim[-2]*dim[-1],axis=1)
            layers_action=Reshape((self._n_actions,-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder)))(layers_action)
            layers_action=Permute((2,3,1), input_shape=(self.n_channels_internal_dim+self._n_actions,-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder)))(layers_action)    #data_format='channels_last'
            
            x = Concatenate(axis=-1)([layers_action,inputs[0]])
            
            x = Conv2D(16, (1, 1), padding='same', activation='tanh')(x) # Try to keep locality as much as possible --> FIXME
            x = Conv2D(32, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(64, (3, 3), padding='same', activation='tanh')(x)
            x = Conv2D(32, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(16, (1, 1), padding='same', activation='tanh')(x)
            #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
            x = Conv2D(self.n_channels_internal_dim, (1, 1), padding='same')(x)
            x = Add()([inputs[0],x])
        else:
            inputs = [ Input( shape=(self.internal_dim,) ), Input( shape=(self._n_actions,) ) ]     # x

            x = Concatenate()(inputs)#,axis=-1)
            x = Dense(10, activation='tanh')(x) #5,15
            x = Dense(30, activation='tanh')(x) # ,30
            x = Dense(30, activation='tanh')(x) # ,30
            x = Dense(10, activation='tanh')(x) # ,30
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

    def diff_Tx_x_(self,encoder_model,transition_model,plan_depth=0):
        """
        Used to fit the transitions
        
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
                if len(dim) == 3 or len(dim) == 4:
                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                    inputs.append(input)
            
                elif len(dim) == 2:
                    input = Input(shape=(dim[-3],dim[-2]))
                    inputs.append(input)
            
                else:
                    input = Input(shape=(dim[-3],))
                    inputs.append(input)

        half = len(inputs)/2
        enc_x = encoder_model(inputs[:half]) #s --> x
        enc_x_ = encoder_model(inputs[half:]) #s --> x

        Tx= enc_x
        for d in range(plan_depth+1):
            inputs.append(Input(shape=(self._n_actions,)))
            Tx= transition_model([Tx,inputs[-1]])
                
        print "Tx._keras_shape"
        print Tx._keras_shape
        print enc_x_._keras_shape
        
        x = Subtract()([Tx,enc_x_])

        input = Input(shape=(1,)) # 1-terminals (0 if transition is terminal)
        inputs.append(input)
        x = Multiply()([x,inputs[-1]])# set to 0 if terminal because we don't care about fitting that transition
        
        model = Model(inputs=inputs, outputs=x )
        
        return model

    def force_features(self,encoder_model,transition_model,plan_depth=0):
        """
        Used to force some transitions'directions
        
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
        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3 or len(dim) == 4:
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
        
        print "Tx._keras_shape"
        print Tx._keras_shape
        
#        input = Input(shape=(self.internal_dim,self._n_actions))
#        inputs.append(input)
#        
#        #if(high_int_dim==True):
#        #    Tx_tiled=K.tile(Tx,(self._n_actions,1,1,1))
#        #else:
#        #    Tx_tiled=K.tile(Tx,(self._n_actions,1))
#        
#        for i in range self._n_actions:            
#            #constants = np.zeros((self._n_actions))
#            #k_constants = K.variable(constants)
#            #fixed_input = Input(tensor=k_constants)
#            Tx= transition_model([Tx,constants])
#        Tx_tiled=Dot(axes=(-1))([Tx,fixed_input])
#
#        print "Tx_tiled._keras_shape"
#        print Tx_tiled._keras_shape
            
        diff_features = Subtract()([Tx,enc_x]) # Modification of the features after (sequence of) action(s)

        #print "K.eval(diff_features)"
        #print diff_features.output
        #inputs.append(Input(shape=(self.internal_dim,)))
        #cos_proxi=Dot(axes=(-1),normalize=True)([diff_features,inputs[-1]]) # Cosine proximity between diff_features and target_modif_features
        
        #constants = np.ones((self.internal_dim,))#((self._batch_size*self._n_actions,self.internal_dim,))
        #k_constants = K.variable(constants)
        #fixed_input = Input(tensor=k_constants)
        #inputs.append(fixed_input)
        #print "fixed_input._keras_shape"
        #print fixed_input._keras_shape
        #cos_proxi_add1=Subtract()([fixed_input,cos_proxi])
        
        #print "cos_proxi.output"
        #print cos_proxi.output
        #print "cos_proxi._keras_shape"
        #print cos_proxi._keras_shape
        
        model = Model(inputs=inputs, outputs=diff_features )
        
        return model


#    def diff_s_s_(self,encoder_model):
#        """
#        Used to force some state representation to be sufficiently different
#        
#        Parameters
#        -----------
#        s
#        a
#        random z
#    
#        Returns
#        -------
#        model with output Tx (= model estimate of x')
#    
#        """
#        inputs=[]
#        
#        for j in range(2):
#            for i, dim in enumerate(self._input_dimensions):
#                if len(dim) == 3:
#                    input = Input(shape=(dim[-3],dim[-2],dim[-1]))
#                    inputs.append(input)
#            
#                elif len(dim) == 2:
#                    input = Input(shape=(dim[-3],dim[-2]))
#                    inputs.append(input)
#            
#                else:
#                    input = Input(shape=(dim[-3],))
#                    inputs.append(input)
#        
#        half = len(inputs)/2
#        enc_x = encoder_model(inputs[:half]) #s --> x #FIXME
#        enc_x_ = encoder_model(inputs[half:]) #s --> x
#        
#        if (self._high_int_dim==True):
#            enc_x=Flatten()(enc_x)
#            enc_x_=Flatten()(enc_x_)
#        x = Subtract()([enc_x,enc_x_])
#
#        #x = Dot(axes=-1, normalize=False)([x,x])
#        
#        model = Model(inputs=inputs, outputs=x )
#        
#        return model

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
            if len(dim) == 3 or len(dim) == 4:
                input = Input(shape=(dim[-3],dim[-2],dim[-1]))
                inputs.append(input)

            elif len(dim) == 2:
                input = Input(shape=(dim[-3],dim[-2]))
                inputs.append(input)

            else:
                input = Input(shape=(dim[-3],))
                inputs.append(input)
        
        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        input = Input(shape=(self._n_actions,))
        inputs.append(input)
        
        enc_x = encoder_model(inputs[:-2]) #s --> x
        Tx= transition_model([enc_x,inputs[-2]])
        rand_Tx= transition_model([enc_x,inputs[-1]])
        
        if (self._high_int_dim==True):
            Tx=Flatten()(Tx)
            rand_Tx=Flatten()(rand_Tx)
            x = Subtract()([Tx,rand_Tx])
        else:
            x = Subtract()([Tx,rand_Tx])
        print "x._keras_shape"
        print x._keras_shape
        #x = Dot(axes=-1, normalize=False)([x,x])
        #print "x._keras_shape"
        #print x._keras_shape
        
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
        
        if(self._high_int_dim==True):
            dim=self._input_dimensions[0] #FIXME
            inputs = [ Input(shape=(-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)), Input( shape=(self._n_actions,) ) ]     #data_format='channels_last'
            
            layers_action=inputs[1]
            layers_action=RepeatVector(-(-dim[-2] // self._pooling_encoder)*-(-dim[-1] // self._pooling_encoder))(layers_action)
            print layers_action._keras_shape
            layers_action=Reshape((self._n_actions,-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder)))(layers_action)
            layers_action=Permute((2,3,1), input_shape=(self.n_channels_internal_dim+self._n_actions,-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder)))(layers_action)    #data_format='channels_last'
            print layers_action._keras_shape

            
            x = Concatenate(axis=-1)([layers_action,inputs[0]])
            x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
            x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
            x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
            #x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
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

    def full_R_model(self,encoder_model,R_model,plan_depth=0,transition_model=None):
        """
        Maps internal state to immediate rewards

        Parameters
        -----------
        s
        a
    
        Returns
        -------
        r
        """
        
        inputs=[]
        
        for i, dim in enumerate(self._input_dimensions):
            if len(dim) == 3 or len(dim) == 4:
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
        
        out = R_model([Tx]+inputs[-1:])

        model = Model(inputs=inputs, outputs=out)
        
        return model

    def Q_model(self):
        if(self._high_int_dim==True):
            inputs=[]
            outs_conv=[]
            for i, dim in enumerate(self._input_dimensions):
                # - observation[i] is a FRAME
                print "dim Q mod"
                print dim
                if len(dim) == 3 or len(dim) == 4:
                    input = Input(shape=(-(-dim[-2] // self._pooling_encoder),-(-dim[-1] // self._pooling_encoder),self.n_channels_internal_dim)) #data_format is already 'channels_last'
                    inputs.append(input)
                    #reshaped=Permute((2,3,1), input_shape=(dim[-3],dim[-2],dim[-1]))(input)
                    x = input     #data_format is already 'channels_last'
                    print x._keras_shape
            
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(32, (3, 3), padding='same', activation='tanh')(x)
                    x = MaxPooling2D(pool_size=(2, 2), strides=None, padding='same')(x)
                    x = Conv2D(16, (2, 2), padding='same', activation='tanh')(x)
                    x = Conv2D(4, (1, 1), padding='same', activation='tanh')(x)
                    out = (x)
                else:
                    print ("FIXME")
                        
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
            x = Flatten()(x)
            x = Dense(200, activation='tanh')(x)
        else:
            inputs = [ Input( shape=(self.internal_dim,) ) ] #x
            x = Dense(20, activation='tanh')(inputs[0])

        
        #if (self._action_as_input==True):
        #    if ( isinstance(self._n_actions,int)):
        #        print("Error, env.nActions() must be a continuous set when using actions as inputs in the NN")
        #    else:
        #        input = Input(shape=(len(self._n_actions),))
        #        inputs.append(input)
                
        #x = Add()([x,inputs[-1]]) #????
        
        # we stack a deep fully-connected network on top
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


    def full_Q_model(self, encoder_model, Q_model, plan_depth=0, transition_model=None, R_model=None, discount_model=None):
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
            if len(dim) == 3 or len(dim) == 4:
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
            print inputs[-1:]
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

        #if(self._high_int_dim==True):
        #    input = Input(shape=(dim[-2],dim[-1],dim[-3]))
        #    inputs.append(input)
        #else:
        #    input = Input(shape=(self.internal_dim,))
        #    inputs.append(input)
        #
        #x=Add()([out,inputs[-1]]) # adding noise in the abstract state space
        
        if(plan_depth==0):
            Q_estim=Q_model(out)
        else:
            Q_estim = Multiply()([disc_plan,Q_model(out)])
            Q_estim = Add()([Q_estim]+disc_rewards)

        model = Model(inputs=inputs, outputs=Q_estim)
        
        return model

if __name__ == '__main__':
    pass
    