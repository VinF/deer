#General_Deep_Q_RL

## Full Documentation

See the [Wiki](https://github.com/VinF/General_Deep_Q_RL/wiki) for full documentation, examples and other information.

## What does it do?

This package provides a Lasagne/Theano-based implementation for a general deep Q-learning algorithm. It is based on the original deep Q learning algorithm described in :

Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

Contrary to the original code, this package provides a more general framework where observations are made up of any number of elements : scalars, vectors and frames (instead of one type of frame only in the above mentionned paper). At each time step, an action is taken by the agent and one observation is gathered. The belief state on which the agent is based to build the Q function is made up of any length history of each element provided in the observation.


## Dependencies

* A reasonably modern NVIDIA GPU
* OpenCV
* [Theano](http://deeplearning.net/software/theano/) ([https://github.com/Theano/Theano](https://github.com/Theano/Theano))
* [Lasagne](http://lasagne.readthedocs.org/en/latest/) ([https://github.com/Lasagne/Lasagne](https://github.com/Lasagne/Lasagne)
