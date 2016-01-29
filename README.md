#General_Deep_Q_RL
## Introduction 

This package provides a Lasagne/Theano-based implementation for a general deep Q-learning algorithm. It is based on the original deep Q learning algorithm described in :

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

Contrary to the original code, this package provides a more general framework where observations are made up of any number of elements : scalars, vectors and frames (instead of one type of frame only in the above mentionned paper). At each time step, an action is taken by the agent and one observation is gathered. The belief state on which the agent is based to build the Q function is made up of any length history of each element provided in the observation.


## Dependencies

* A reasonably modern NVIDIA GPU
* OpenCV
* [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
* [Lasagne](http://lasagne.readthedocs.org/en/latest/) ([https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne)


## How to use it - Examples

### Toy example
The first example can be run by using 
```
python run_toy_env
```
This example uses the environment defined in Toy_env.py. It consists in buying or selling goods given a price pattern. The price pattern is made by repeating the following signal plus a random constant between 0 and 3:
<img src="/Images/plot_toy_example_signal.png" height="200" width="200" alt="Toy example">

After ten epochs, the following graph is obtained:
<img src="/Images/plot_toy_example.png" height="200" width="200" alt="Toy example">

In this graph, you can see that the agent has successfully learned after 10 epochs to take advantage of the price pattern to buy when it is low and to sell when it is high. This example is of course easy due to the fact that the patterns are very systematic which allows the agent to successfuly learn it. It is important to note that the results shown are made on a validation set that is different from the training and you can see the expected behaviour of buying at time step 7 and 16 (Because in average this will allow to make profit)



### Two storage devices example
This second example is slightly more complex and realistic. It simulates the operation of a micro-grid (such as a smart home for instance) that is not connected to the main utility grid (off-grid) and that is provided with PV panels, batteries and hydrogen storage. The battery has the advantage that it is not limited in instaneous power that it can provide or store. The hydrogen storage has the advantage that is can store very large quantity of energy. Details can be found in the MG_two_storage_devices_env.py.

```
python run_MG_two_storage_devices
```



