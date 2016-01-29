"""This class stores all of the samples for training. 

Author: Vincent Francois-Lavet
"""

import numpy as np
import theano
import copy 


floatX = theano.config.floatX

class DataSet(object):
    """A replay memory consisting of circular buffers for observed images,
actions, and rewards.

    """
    def __init__(self, num_elements_in_batch, rng, max_steps=1000):
        """Construct a DataSet.

        Arguments:
            num_elements_in_batch - 
            rng - initialized numpy random number generator, used to
            max_steps - 

        """
        self.num_elements_in_batch = num_elements_in_batch
        
        self.max_steps = max_steps
        self.rng = rng

        self.element=[]
        self.elements=[]

        self.actions = np.zeros(max_steps, dtype='int32')
        self.rewards = np.zeros(max_steps, dtype=floatX)
        self.terminals = np.zeros(max_steps, dtype='bool')

        self.bottom = 0
        self.top = 0
        self.size = 0


    def add_sample_1(self, sample):
        """Add the element in the time step record.

        Arguments:
            img -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.element=sample

        if self.size < self.max_steps:
            self.elements.append(copy.copy(self.element))
        else:
            self.elements[self.top]=copy.copy(self.element)
        

        if self.size == self.max_steps:
            self.bottom = (self.bottom + 1) % self.max_steps
        else:
            self.size += 1
            
        self.top = (self.top + 1) % self.max_steps

        return self.top-1


    def add_sample_2(self, index, action, reward, terminal):
        """Add the action taken, the reward and whether this is the end of an episode in the time step record.

        Arguments:
            index -- 
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended after this time step
        """
        self.rewards[index] = reward
        self.terminals[index] = terminal
        self.actions[index] = action

    def __len__(self):
        """Return the count of stored state transitions."""
        # TODO: Properly account for indices which can't be used, as in
        # random_batch's check.
        return self.size



    def random_batch(self, batch_size):
        """Return corresponding states, actions, rewards, terminal status, and
next_states for batch_size randomly chosen state transitions.
        
        Arguments:
            batch_size - int, number of elements in the batch

        Returns:
            states - list of batch_size elements (element = [list of k*[list of max_num_elements*[element 2D,1D or scalar]])
            actions, 
            rewards, 
            next_states, 
            terminal    

        """
        # Allocate the response.
        actions = np.zeros((batch_size, 1), dtype='int32')
        rewards = np.zeros((batch_size, 1), dtype=floatX)
        terminal = np.zeros((batch_size, 1), dtype='bool')

        states=[]
        next_states=[]
        count = 0
        while count < batch_size:
        
            # Randomly choose a time step from the replay memory.             
            if (self.bottom + max(self.num_elements_in_batch) < self.size-1):
                end_index1 =  self.rng.randint(self.bottom + max(self.num_elements_in_batch),
                                     self.size-1)
            else:
                end_index1=0
            if (max(self.num_elements_in_batch) < self.top-1):
                end_index2 =  self.rng.randint(max(self.num_elements_in_batch),
                                     self.top-1)
            else:
                end_index2=0

            
            if (end_index1==0):
                end_index=end_index2
            elif (end_index2==0):
                end_index=end_index1
            else:
                end_index=self.rng.choice([end_index1,end_index2]) # FIXME --> part of the index set not considered while could be!
                            
            actions[count] = self.actions.take(end_index, mode='wrap')
            rewards[count] = self.rewards.take(end_index, mode='wrap')
            terminal[count] = self.terminals.take(end_index, mode='wrap')

            states_t = self.get_one_state(end_index)
            next_states_t= self.get_one_state(end_index+1)
            
            states.append(states_t)
            next_states.append(next_states_t)
            
            count += 1

        return states, actions, rewards, next_states, terminal



    def get_one_state(self,end_index):
        states_t=[]
            
        for i,index in enumerate( np.arange(end_index,end_index-max(self.num_elements_in_batch),-1 ) ):
            if index<0:
                index=index+self.size
            the_state=[]

            for j,element in enumerate( zip(self.elements[index]) ):#,self.elements[(index+1)%self.size]) ):
                if(i<self.num_elements_in_batch[j]):
                    the_state.append(element[0])
                else:
                    the_state.append(None)
            
            states_t.append(the_state)
            
        return states_t
        
    


if __name__ == "__main__":
    pass
