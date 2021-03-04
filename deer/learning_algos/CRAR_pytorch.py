"""
Code for the CRAR learning algorithm using Keras

"""

import numpy as np
np.set_printoptions(threshold=np.nan)
from keras.optimizers import SGD,RMSprop
from keras import backend as K
from ..base_classes import LearningAlgo
from .NN_CRAR_pytorch import NN # Default Neural network used
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
import copy
import torch
import torch.nn.functional as F
import torch.optim as optim
import pdb 

def mean_squared_error_p(y_true, y_pred):
    """ Modified mean square error that clips
    """
    return K.clip(K.max(  K.square( y_pred - y_true )  ,  axis=-1  )-1,0.,100.)     # = modified mse error L_inf
    #return K.clip(K.mean(  K.square( y_pred - y_true )  ,  axis=-1  )-1,0.,100.)   # = modified mse error L_2


def mean_squared_error_p_pytorch(y_pred):
    """ Modified mean square error that clips
    """
    return  torch.sum(torch.clamp( (torch.max((y_pred)**2,dim=-1)[0] - 1), 0., 100.)) # = modified mse error L_inf

def exp_dec_error(y_true, y_pred):
    return K.exp( - 5.*K.sqrt( K.clip(K.sum(K.square(y_pred), axis=-1, keepdims=True),0.000001,10) )  ) 


def exp_dec_error_pytorch(y_pred):
    return torch.mean(torch.exp( - 5.*torch.sqrt( torch.clamp(torch.sum(y_pred**2, dim=-1),0.000001,10) )  ))



def cosine_proximity2(y_true, y_pred):
    """ This loss is similar to the native cosine_proximity loss from Keras
    but it differs by the fact that only the two first components of the two vectors are used
    """
    y_true = K.l2_normalize(y_true[:,0:2], axis=-1)
    y_pred = K.l2_normalize(y_pred[:,0:2], axis=-1)
    return -K.sum(y_true * y_pred, axis=-1)    


def cosine_proximity2_pytorch(y_true, y_pred):
    """ This loss is similar to the native cosine_proximity loss from Keras
    but it differs by the fact that only the two first components of the two vectors are used
    """

    y_true = F.normalize(y_true[:,0:2],p=2,dim=-1)
    y_pred = F.normalize(y_pred[:,0:2],p=2,dim=-1)
    return -torch.sum(y_true * y_pred, dim=-1)


# def loss_diff_s_s_(y_true, y_pred):
#     return K.square(   1.    -    K.sqrt(  K.clip( K.sum(y_pred,axis=-1,keepdims=True), 0.000001 , 1. )  )     ) # tend to increase y_pred --> loss -1

class CRAR(LearningAlgo):
    """
    Combined Reinforcement learning via Abstract Representations (CRAR) using Keras
    
    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent evolves.
    rho : float
        Parameter for rmsprop. Default : 0.9
    rms_epsilon : float
        Parameter for rmsprop. Default : 0.0001
    momentum : float
        Momentum for SGD. Default : 0
    clip_norm : float
        The gradient tensor will be clipped to a maximum L2 norm given by this value.
    freeze_interval : int
        Period during which the target network is freezed and after which the target network is updated. Default : 1000
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    update_rule: str
        {sgd,rmsprop}. Default : rmsprop
    random_state : numpy random number generator
        Set the random seed.
    double_Q : bool, optional
        Activate or not the double_Q learning.
        More informations in : Hado van Hasselt et al. (2015) - Deep Reinforcement Learning with Double Q-learning.
    neural_network : object, optional
        Default is deer.learning_algos.NN_keras
    """

    def __init__(self, environment, rho=0.9, rms_epsilon=0.0001, momentum=0, clip_norm=0, freeze_interval=1000, batch_size=32, update_rule="rmsprop", random_state=np.random.RandomState(), double_Q=False, neural_network=NN, **kwargs):
        """ Initialize the environment
        
        """
        LearningAlgo.__init__(self,environment, batch_size)

        self._rho = rho
        self._rms_epsilon = rms_epsilon
        self._momentum = momentum
        self._clip_norm = clip_norm
        self._update_rule = update_rule
        self._freeze_interval = freeze_interval
        self._double_Q = double_Q
        self._random_state = random_state
        self.update_counter = 0    
        self._high_int_dim = kwargs.get('high_int_dim',False)
        self._internal_dim = kwargs.get('internal_dim',2)
        self.loss_interpret=0
        self.loss_T=0
        self.lossR=0
        self.loss_Q=0
        self.loss_disentangle_t=0
        self.loss_disambiguate1=0
        self.loss_disambiguate2=0
        self.loss_gamma=0
        
        self.learn_and_plan = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, high_int_dim=self._high_int_dim, internal_dim=self._internal_dim)


        self.encoder = self.learn_and_plan.encoder_model()
        self.encoder_diff = self.learn_and_plan.encoder_diff_model

        
        self.R = self.learn_and_plan.float_model()
        self.Q = self.learn_and_plan.Q_model()
        self.gamma = self.learn_and_plan.float_model()
        self.transition = self.learn_and_plan.transition_model()

        self.full_Q=self.learn_and_plan.full_Q_model

        
        # used to fit rewards
        self.full_R = self.learn_and_plan.full_float_model


        # used to fit gamma
        self.full_gamma = self.learn_and_plan.full_float_model

        
        # used to fit transitions
        self.diff_Tx_x_ = self.learn_and_plan.diff_Tx_x_

        # constraint on consecutive t
        self.diff_s_s_ = self.learn_and_plan.encoder_diff_model


        # used to force features variations
        if(self._high_int_dim==False):
            self.force_features=self.learn_and_plan.force_features


        self.all_models = [self.encoder,self.R,self.Q,self.gamma,self.transition]

        # Compile all models
        self._compile()

        
        # Instantiate the same neural network as a target network.
        self.learn_and_plan_target = neural_network(self._batch_size, self._input_dimensions, self._n_actions, self._random_state, high_int_dim=self._high_int_dim, internal_dim=self._internal_dim)
        self.encoder_target = self.learn_and_plan_target.encoder_model()
        self.Q_target = self.learn_and_plan_target.Q_model()
        self.R_target = self.learn_and_plan_target.float_model()
        self.gamma_target = self.learn_and_plan_target.float_model()
        self.transition_target = self.learn_and_plan_target.transition_model()

        self.full_Q_target = self.learn_and_plan_target.full_Q_model


        self.all_models_target = [self.encoder_target,self.R_target,self.Q_target,self.gamma_target,self.transition_target]

        self._resetQHat()



    def train(self, states_val, actions_val, rewards_val, next_states_val, terminals_val):
        """
        Train CRAR from one batch of data.

        Parameters
        -----------
        states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        actions_val : numpy array of integers with size [self._batch_size]
            actions[i] is the action taken after having observed states[:][i].
        rewards_val : numpy array of floats with size [self._batch_size]
            rewards[i] is the reward obtained for taking actions[i-1].
        next_states_val : numpy array of objects
            Each object is a numpy array that relates to one of the observations
            with size [batch_size * history size * size of punctual observation (which is 2D,1D or scalar)]).
        terminals_val : numpy array of booleans with size [self._batch_size]
            terminals[i] is True if the transition leads to a terminal state and False otherwise

        Returns
        -------
        Average loss of the batch training for the Q-values (RMSE)
        Individual (square) losses for the Q-values for each tuple
        """
        
        onehot_actions = np.zeros((self._batch_size, self._n_actions))
        onehot_actions[np.arange(self._batch_size), actions_val] = 1
        onehot_actions_rand = np.zeros((self._batch_size, self._n_actions))
        onehot_actions_rand[np.arange(self._batch_size), np.random.randint(0,2,(32))] = 1
        states_val=list(states_val)
        next_states_val=list(next_states_val)
        
        
        states_val = torch.from_numpy(states_val[0]).float()
        next_states_val = torch.from_numpy(next_states_val[0]).float()
        onehot_actions = torch.from_numpy(onehot_actions).float()
        terminals_val = torch.from_numpy(terminals_val[:,None].astype(np.int32)).float()
        rewards_val = torch.from_numpy(rewards_val[:,None].astype(np.int32)).float()

        Es_=self.encoder.predict(next_states_val).detach()
        Es=self.encoder.predict(states_val).detach()
        ETs=self.transition.predict(torch.cat((Es,onehot_actions),-1)).detach()
        R=self.R.predict(torch.cat((Es,onehot_actions),-1)).detach()
                   
        if(self.update_counter%500==0):
            print ("Printing a few elements useful for debugging:")
            #print ("states_val[0][0]")
            #print (states_val[0][0])
            #print ("next_states_val[0][0]")
            #print (next_states_val[0][0])
            print ("actions_val[0], rewards_val[0], terminals_val[0]")
            print (actions_val[0], rewards_val[0], terminals_val[0])
            print ("Es[0],ETs[0],Es_[0]")

            # if(Es.ndim==4):
            #     print (np.transpose(Es, (0, 3, 1, 2))[0],np.transpose(ETs, (0, 3, 1, 2))[0],np.transpose(Es_, (0, 3, 1, 2))[0])    # data_format='channels_last' --> 'channels_first'
            # else:
            print (Es[0],ETs[0],Es_[0])
            print ("R[0]")
            print (R[0])
        
        self.optimizer_diff_Tx_x_.zero_grad()
        out = self.diff_Tx_x_(states_val,next_states_val,onehot_actions,(1-terminals_val),self.encoder,self.transition)
        loss = torch.nn.MSELoss()
        loss_val = loss(out,torch.zeros_like(Es))
        self.loss_T+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.transition.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_diff_Tx_x_.step()
        
        

        self.optimizer_full_R.zero_grad()
        out = self.full_R(states_val,onehot_actions,self.encoder,self.R)
        loss = torch.nn.MSELoss()
        loss_val = loss(out,rewards_val)
        self.lossR+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.encoder.parameters()) + list(self.R.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_full_R.step()


        self.optimizer_full_gamma.zero_grad()
        out = self.full_gamma(states_val,onehot_actions,self.encoder,self.gamma)
        loss = torch.nn.MSELoss()
        loss_val = loss(out,(1-terminals_val[:])*self._df)
        self.loss_gamma+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.encoder.parameters()) + list(self.gamma.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_full_gamma.step()   
        
        

        L_infinity ball of radius 1 loss
        self.optimizer_encoder.zero_grad()
        out = self.encoder(states_val)
        loss_val = mean_squared_error_p_pytorch(out)
        self.loss_disambiguate1+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.encoder.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_encoder.step()




        # This one is very important
        # Entropy maximization loss (through exponential) between two random states
        def roll(x, n):  
            return torch.cat((x[-n:], x[:-n]))        
        rolled = roll(states_val,-31)
        self.optimizer_encoder_diff.zero_grad()
        out = self.encoder_diff(self.encoder,states_val,rolled)
        loss_val = exp_dec_error_pytorch(out)

        self.loss_disambiguate2+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.encoder.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_encoder_diff.step()
    



        # Not so much this one
        # Entropy maximization loss (through exponential) between two consecutive states
        self.optimizer_diff_s_s_.zero_grad()
        out = self.diff_s_s_(self.encoder,states_val,next_states_val)
        loss_val = exp_dec_error_pytorch(out)
        self.loss_disentangle_t+= loss_val.data.numpy()
        loss_val.backward()
        for param in list(self.encoder.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_diff_s_s_.step()
 

        # Q Learning loss
        if self.update_counter % self._freeze_interval == 0:
            self._resetQHat()
        next_q_vals = self.full_Q_target(next_states_val,self.encoder_target,self.Q_target).detach()
        max_next_q_vals=torch.max(next_q_vals, dim=1)[0]
        not_terminals= (1 - terminals_val)
        target = rewards_val + not_terminals * self._df * max_next_q_vals[:,None]
   
        self.optimizer_full_Q.zero_grad()
        q_vals=self.full_Q(states_val,self.encoder,self.Q).gather(1, torch.from_numpy(actions_val.astype(int)[:,None]))
        loss = torch.nn.MSELoss()
        loss_val = loss(q_vals,target)
        loss = loss_val.data.numpy()
        self.loss_Q+= loss
        loss_val.backward()
        for param in list(self.encoder.parameters()) + list(self.Q.parameters()):
            param.grad.data.clamp_(-1, 1)
        self.optimizer_full_Q.step()



    

        
        if(self.update_counter%500==0):
            print ("self.loss_T/500., self.lossR/500., self.loss_gamma/500., self.loss_Q/500., self.loss_disentangle_t/500., self.loss_disambiguate1/500., self.loss_disambiguate2/500.")
            print (self.loss_T/500., self.lossR/500.,self.loss_gamma/500., self.loss_Q/500., self.loss_disentangle_t/500., self.loss_disambiguate1/500., self.loss_disambiguate2/500.)

            if(self._high_int_dim==False):
                print ("self.loss_interpret/500.")
                print (self.loss_interpret/500.)

            self.lossR=0
            self.loss_gamma=0
            self.loss_Q=0
            self.loss_T=0
            self.loss_interpret=0

            self.loss_disentangle_t=0
            self.loss_disambiguate1=0
            self.loss_disambiguate2=0


        if(self.update_counter%100==0):
            print ("Number of training steps:"+str(self.update_counter)+".")
        
        self.update_counter += 1           




        return np.sqrt(loss),(q_vals.detach()-target)**2

  
    def _compile(self):
        """ Compile all the optimizers for the different losses
        """
        

        if (self._update_rule=="rmsprop"):
            self.optimizer_full_Q=optim.RMSprop(list(self.encoder.parameters()) + list(self.Q.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
            self.optimizer_diff_Tx_x_=optim.RMSprop( list(self.encoder.parameters()) +list(self.transition.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon) # Different optimizers for each network; 
            self.optimizer_full_R=optim.RMSprop(list(self.encoder.parameters()) + list(self.R.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon) # to possibly modify them separately
            self.optimizer_full_gamma=optim.RMSprop(list(self.encoder.parameters()) + list(self.gamma.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon) 
            self.optimizer_encoder=optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
            self.optimizer_encoder_diff=optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
            self.optimizer_diff_s_s_=optim.RMSprop(self.encoder.parameters(), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon)
            # self.optimizer_force_features=optim.RMSprop(list(self.encoder.parameters()) + list(self.transition.parameters()), lr=self._lr, alpha=self._rho, eps=self._rms_epsilon) # This never gets updated

        else:
            raise Exception('The update_rule '+self._update_rule+' is not implemented.')
        
        self.optimizers = [self.optimizer_full_Q,self.optimizer_diff_Tx_x_,
                           self.optimizer_full_R,self.optimizer_full_gamma,
                           self.optimizer_encoder,self.optimizer_encoder_diff,
                           self.optimizer_diff_s_s_  ]


    def qValues(self, state_val):
        """ Get the q values for one pseudo-state (without planning)

        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).

        Returns
        -------
        The q values for the provided pseudo state
        """ 
        copy_state=copy.deepcopy(state_val) #Required!

        return self.full_Q.predict([np.expand_dims(state,axis=0) for state in copy_state])[0]

    def qValues_planning(self, state_val, R, gamma, T, Q, d=5):
        """ Get the average Q-values up to planning depth d for one pseudo-state.
        
        Arguments
        ---------
        state_val : array of objects (or list of objects)
            Each object is a numpy array that relates to one of the observations
            with size [1 * history size * size of punctual observation (which is 2D,1D or scalar)]).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The average q values with planning depth up to d for the provided pseudo-state
        """
        encoded_x = self.encoder.predict(state_val)

#        ## DEBUG PURPOSES
#        print ( "self.full_Q.predict(state_val)[0]" )
#        print ( self.full_Q.predict(state_val)[0] )
#        identity_matrix = np.diag(np.ones(self._n_actions))
#        if(encoded_x.ndim==2):
#            tile3_encoded_x=np.tile(encoded_x,(self._n_actions,1))
#        elif(encoded_x.ndim==4):
#            tile3_encoded_x=np.tile(encoded_x,(self._n_actions,1,1,1))
#        else:
#            print ("error")
#        
#        repeat_identity=np.repeat(identity_matrix,len(encoded_x),axis=0)
#        ##print tile3_encoded_x
#        ##print repeat_identity
#        r_vals_d0=np.array(R.predict([tile3_encoded_x,repeat_identity]))
#        #print "r_vals_d0"
#        #print r_vals_d0
#        r_vals_d0=r_vals_d0.flatten()
#        print "r_vals_d0"
#        print r_vals_d0
#        next_x_predicted=T.predict([tile3_encoded_x,repeat_identity])
#        #print "next_x_predicted"
#        #print next_x_predicted
#        one_hot_first_action=np.zeros((1,self._n_actions))
#        one_hot_first_action[0]=1
#        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
#        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
#        next_x_predicted=T.predict([next_x_predicted[0:1],one_hot_first_action])
#        #print "next_x_predicted action 0 t4"
#        #print next_x_predicted
#        ## END DEBUG PURPOSES

        QD_plan=0
        for i in range(d+1):
            Qd=self.qValues_planning_abstr(encoded_x, R, gamma, T, Q, d=i, branching_factor=[self._n_actions,2,2,2,2,2,2,2]).reshape(len(encoded_x),-1)
            print ("Qd,i")
            print (Qd,i)
            QD_plan+=Qd
        QD_plan=QD_plan/(d+1)
        
        print ("QD_plan")
        print (QD_plan)

        return QD_plan
  
    def qValues_planning_abstr(self, state_abstr_val, R, gamma, T, Q, d, branching_factor=None):
        """ Get the q values for pseudo-state(s) with a planning depth d. 
        This function is called recursively by decreasing the depth d at every step.

        Arguments
        ---------
        state_abstr_val : internal state(s).
        R : float_model
            Model that fits the reward
        gamma : float_model
            Model that fits the discount factor
        T : transition_model
            Model that fits the transition between abstract representation
        Q : Q_model
            Model that fits the optimal Q-value
        d : int
            planning depth

        Returns
        -------
        The Q-values with planning depth d for the provided encoded state(s)
        """
        #if(branching_factor==None or branching_factor>self._n_actions):
        #    branching_factor=self._n_actions
        


        n=len(state_abstr_val)
        identity_matrix = np.identity(self._n_actions)
        
        this_branching_factor=branching_factor.pop(0)
        if (n==1):
            # We require that the first branching factor is self._n_actions so that this function return values 
            # with the right dimension (=self._n_actions). 
            this_branching_factor=self._n_actions
                         
        if (d==0):
            if(this_branching_factor<self._n_actions):
                return np.partition(Q.predict([state_abstr_val]), -this_branching_factor)[:,-this_branching_factor:]
            else:
                return Q.predict([state_abstr_val]) # no change in the order of the actions
        else:
            if(this_branching_factor==self._n_actions):
                # All actions are considered in the tree
                # NB: For this case, we do not use argpartition because we want to keep the actions in the natural order
                # That way, this function returns the Q-values for all actions with planning depth d in the right order
                repeat_identity=np.repeat(identity_matrix,len(state_abstr_val),axis=0)
                if(state_abstr_val.ndim==2):
                    tile3_encoded_x=np.tile(state_abstr_val,(self._n_actions,1))
                elif(state_abstr_val.ndim==4):
                    tile3_encoded_x=np.tile(state_abstr_val,(self._n_actions,1,1,1))
                else:
                    print ("error")
            else:
                # A subset of the actions corresponding to the best estimated Q-values are considered et each branch 
                estim_Q_values=Q.predict([state_abstr_val])
                ind = np.argpartition(estim_Q_values, -this_branching_factor)[:,-this_branching_factor:]
                # Replacing ind if we want random branching
                #ind = np.random.randint(0,self._n_actions,size=ind.shape)
                repeat_identity=identity_matrix[ind].reshape(n*this_branching_factor,self._n_actions)
                tile3_encoded_x=np.repeat(state_abstr_val,this_branching_factor,axis=0)
            
            r_vals_d0=np.array(R.predict([tile3_encoded_x,repeat_identity]))
            r_vals_d0=r_vals_d0.flatten()
            
            gamma_vals_d0=np.array(gamma.predict([tile3_encoded_x,repeat_identity]))
            gamma_vals_d0=gamma_vals_d0.flatten()

            next_x_predicted=T.predict([tile3_encoded_x,repeat_identity])
            return r_vals_d0+gamma_vals_d0*np.amax(self.qValues_planning_abstr(next_x_predicted,R,gamma,T,Q,d=d-1,branching_factor=branching_factor).reshape(len(state_abstr_val)*this_branching_factor,branching_factor[0]),axis=1).flatten()

    def chooseBestAction(self, state, mode, *args, **kwargs):
        """ Get the best action for a pseudo-state

        Arguments
        ---------
        state : list of numpy arrays
             One pseudo-state. The number of arrays and their dimensions matches self.environment.inputDimensions().
        mode : int
            Identifier of the mode (-1 is reserved for the training mode).

        Returns
        -------
        The best action : int
        """

        copy_state=copy.deepcopy(state) #Required because of the "hack" below

        if(mode==None):
            mode=0
        di=[0,1,3,6]
        # We use the mode to define the planning depth
        q_vals = self.qValues_planning([np.expand_dims(s,axis=0) for s in copy_state],self.R,self.gamma, self.transition, self.Q, d=di[mode])

        return np.argmax(q_vals),np.max(q_vals)
      
    def _resetQHat(self):
        """ Set the target Q-network weights equal to the main Q-network weights
        """

        # for i,(param,next_param) in enumerate(zip(self.params, self.params_target)):
        #     K.set_value(next_param,K.get_value(param))

        for mod,mod_t in zip(self.all_models,self.all_models_target):
            mod_t.load_state_dict(mod.state_dict())
            mod_t.eval()

    def setLearningRate(self, lr):
        """ Setting the learning rate

        Parameters
        -----------
        lr : float
            The learning rate that has to be set
        """
        
        self._lr = lr
        print ("New learning rate set to "+str(self._lr)+".")
        for i,optim in enumerate(self.optimizers):
            for param_group in optim.param_groups:
                param_group['lr'] = lr if i != len(self.optimizers)-1 else lr/5.



    # Not implemented yet
    # def transfer(self, original, transfer, epochs=1):

    #     # First, make sure that the target network and the current network are the same
    #     self._resetQHat()
    #     # modify the loss of the encoder
    #     optimizer4=RMSprop(lr=self._lr, rho=0.9, epsilon=1e-06)
    #     self.encoder.compile(optimizer=optimizer4, loss='mse')
        
    #     # Then, train the encoder such that the original and transfer states are mapped into the same abstract representation
    #     x_original=self.encoder.predict(original)#[0]
    #     print ("x_original[0:10]")
    #     print (x_original[0:10])
    #     for i in range(epochs):
    #         size = original[0].shape[0]
    #         print ( "train" )
    #         print ( self.encoder.train_on_batch(transfer[0][0:int(size*0.8)] , x_original[0:int(size*0.8)] ) )
    #         print ( "validation" )
    #         print ( self.encoder.test_on_batch(transfer[0][int(size*0.8):] , x_original[int(size*0.8):]) )
         
    #     self.encoder.compile(optimizer=optimizer4,
    #               loss=mean_squared_error_p)

