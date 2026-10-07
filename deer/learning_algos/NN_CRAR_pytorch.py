"""
CRAR Neural network using Keras

"""

import numpy as np
from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer, Dense, Flatten, Activation, Conv2D, MaxPooling2D, UpSampling2D, Reshape, Permute, Add, Subtract, Dot, Multiply, Average, Lambda, Concatenate, BatchNormalization, merge, RepeatVector, AveragePooling2D
from keras import regularizers
#np.random.seed(111111)
import pdb

import torch.nn as nn
import torch.nn.functional as F
import torch


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


        self._pooling_encoder=1
        class Encoder(nn.Module):
            def __init__(self,internal_dim,input_dim):
                super(Encoder, self).__init__()
                self.input_dim_flat = np.prod(input_dim)
                self.lin1 = nn.Linear(self.input_dim_flat, 200)
                self.lin2 = nn.Linear(200, 100)
                self.lin3 = nn.Linear(100, 50)
                self.lin4 = nn.Linear(50, 10)
                self.lin5 = nn.Linear(10, internal_dim)


            def forward(self, x):
                # pdb.set_trace()
                x = x.view(-1, self.input_dim_flat)
                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = torch.tanh(self.lin4(x))
                x = torch.tanh(self.lin5(x))
                # x = self.lin5(x)
                return x

            def predict(self, x):
                return self.forward(x)

        model = Encoder(self.internal_dim,self._input_dimensions)
        
        return model

    def encoder_diff_model(self,encoder_model,s1,s2):
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


        enc_s1= encoder_model(s1)
        enc_s2= encoder_model(s2)

        
        return enc_s1 - enc_s2

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

        # MLP Transition model
        class Transition(nn.Module):
            def __init__(self,internal_dim,n_actions):
                super(Transition, self).__init__()
                self.lin1 = nn.Linear(internal_dim+n_actions, 10)
                self.lin2 = nn.Linear(10, 30)
                self.lin3 = nn.Linear(30, 30)
                self.lin4 = nn.Linear(30, 10)
                self.lin5 = nn.Linear(10, internal_dim)

                self.internal_dim = internal_dim

            def forward(self, x):
                init_state = x[:,:self.internal_dim]
                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = torch.tanh(self.lin4(x))
                x = self.lin5(x)
                return x + init_state

            def predict(self, x):
                return self.forward(x)





        class MLP(nn.Module):
            """Two-layer fully-connected ELU net with batch norm."""

            def __init__(self, n_in, n_hid, n_out, do_prob=0.):
                super(MLP, self).__init__()
                self.fc1 = nn.Linear(n_in, n_hid)
                self.fc2 = nn.Linear(n_hid, n_out)
                # self.bn = nn.BatchNorm1d(n_out)
                self.dropout_prob = do_prob

                self.init_weights()

            def init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight.data)
                        m.bias.data.fill_(0.1)
                    elif isinstance(m, nn.BatchNorm1d):
                        m.weight.data.fill_(1)
                        m.bias.data.zero_()

            def batch_norm(self, inputs):
                x = inputs.view(inputs.size(0) * inputs.size(1), -1)
                x = self.bn(x)
                return x.view(inputs.size(0), inputs.size(1), -1)

            def forward(self, inputs):
                # Input shape: [num_sims, num_things, num_features]
                x = F.elu(self.fc1(inputs))
                x = F.dropout(x, self.dropout_prob, training=self.training)
                x = F.elu(self.fc2(x))
                return x


        # GNN Transition model
        class TransitionGNN(nn.Module):
            def __init__(self, internal_dim, n_actions, n_hid, do_prob=0., factor=True):
                super(TransitionGNN, self).__init__()

                self.internal_dim = internal_dim
                self.n_actions =n_actions

                n_in = 1
                n_out = internal_dim

                self.mlp1 = MLP(n_in, n_hid, n_hid, do_prob)
                self.mlp2 = MLP(n_hid * 2, n_hid, n_hid, do_prob)
                self.mlp3 = MLP(n_hid, n_hid, n_hid, do_prob)
                # self.mlp4 = MLP(n_hid * 4, n_hid, n_hid, do_prob)
                # self.mlp5 = MLP(n_hid, n_hid, n_hid, do_prob)
                self.fc_out1 = nn.Linear(n_hid*2  * (internal_dim+n_actions), n_hid)
                self.fc_out2 = nn.Linear(n_hid, n_out)
                self.init_weights()

                def encode_onehot(labels):
                    classes = set(labels)
                    classes_dict = {c:  np.identity(len(classes))[i, :] for i, c in
                                    enumerate(classes)}
                    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                             dtype=np.int32)
                    return labels_onehot

                off_diag = np.ones([self.internal_dim+self.n_actions, self.internal_dim+self.n_actions]) - np.eye(self.internal_dim+self.n_actions)
                rel_rec = np.array(encode_onehot(np.where(off_diag)[1]), dtype=np.float32)
                rel_send = np.array(encode_onehot(np.where(off_diag)[0]), dtype=np.float32)
                self.rel_rec = torch.FloatTensor(rel_rec)
                self.rel_send = torch.FloatTensor(rel_send)


            def init_weights(self):
                for m in self.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_normal(m.weight.data)
                        m.bias.data.fill_(0.1)

            def edge2node(self, x):
                # NOTE: Assumes that we have the same graph across all samples.
                incoming = torch.matmul(self.rel_rec.t(), x)
                return incoming / incoming.size(1)

            def node2edge(self, x):
                # NOTE: Assumes that we have the same graph across all samples.
                receivers = torch.matmul(self.rel_rec, x)
                senders = torch.matmul(self.rel_send, x)
                edges = torch.cat([receivers, senders], dim=2)
                return edges

            def forward(self, inputs):
                # import pdb;pdb.set_trace()
                
                init_state = inputs[:,:self.internal_dim]
                x = inputs.view(inputs.size(0), inputs.size(1), 1)
                x = self.mlp1(x)  # 2-layer ELU net per node
                x_skip = x

                x = self.node2edge(x)
                x = self.mlp2(x)
                
                x = self.edge2node(x)
                x = self.mlp3(x)

                x = torch.cat((x, x_skip), dim=2)



                x = x.view(x.size(0), -1)
                x= F.elu(self.fc_out1(x))
                x= self.fc_out2(x)
                return x + init_state

            def predict(self, x):
                return self.forward(x)


        # model = Transition(self.internal_dim,self._n_actions)
        model = TransitionGNN(self.internal_dim, self._n_actions, 32)



        return model

    def diff_Tx_x_(self,s1,s2,action,not_terminal,encoder_model,transition_model,plan_depth=0):
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


        enc_s1 = encoder_model(s1)
        enc_s2 = encoder_model(s2)

        Tx = transition_model(torch.cat((enc_s1,action),-1))


        return (Tx - enc_s2)*(not_terminal)

    def force_features(self,s1,s2,action,encoder_model,transition_model,plan_depth=0):
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


        enc_s1 = encoder_model(s1)
        enc_s2 = encoder_model(s2)

        Tx = transition_model(torch.cat((enc_s1,action),-1))


        return (Tx - enc_s2)


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
        

        class FloatModel(nn.Module):
            def __init__(self,internal_dim,n_actions):
                super(FloatModel, self).__init__()
                self.lin1 = nn.Linear(internal_dim+n_actions, 10)
                self.lin2 = nn.Linear(10, 50)
                self.lin3 = nn.Linear(50, 20)
                self.lin4 = nn.Linear(20, 1)

            def forward(self, x):

                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = self.lin4(x)
                return x
            def predict(self, x):
                return self.forward(x)
        model = FloatModel(self.internal_dim,self._n_actions)



        return model

    def full_float_model(self,x,action,encoder_model,float_model,plan_depth=0,transition_model=None):
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
        

        enc_x = encoder_model(x)
        reward_pred = float_model(torch.cat((enc_x,action),-1))
        return reward_pred

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



        class QFunction(nn.Module):
            def __init__(self,internal_dim,n_actions):
                super(QFunction, self).__init__()
                self.lin1 = nn.Linear(internal_dim, 20)
                self.lin2 = nn.Linear(20, 50)
                self.lin3 = nn.Linear(50, 20)
                self.lin4 = nn.Linear(20, n_actions)

            def forward(self, x):
                x = torch.tanh(self.lin1(x))
                x = torch.tanh(self.lin2(x))
                x = torch.tanh(self.lin3(x))
                x = self.lin4(x)
                return x
            def predict(self, x):
                return self.forward(x)

        model = QFunction(self.internal_dim,self._n_actions)  




        return model


    def full_Q_model(self, x, encoder_model, Q_model, plan_depth=0, transition_model=None, R_model=None, discount_model=None):
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
        
        out = encoder_model(x)
        Q_estim= Q_model(out)

        return Q_estim

if __name__ == '__main__':
    pass
    