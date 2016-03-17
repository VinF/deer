#General_Deep_Q_RL

## Full Documentation

See the [Wiki](https://github.com/VinF/General_Deep_Q_RL/wiki) for full documentation, examples and other information.

## What does it do?

This package provides a general deep Q-learning algorithm. It is based on the original deep Q learning algorithm described in :
Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

Contrary to the original code, this package provides a more general framework where observations are made up of any number of elements : scalars, vectors and frames (instead of one type of frame only in the above mentionned paper). At each time step, an action is taken by the agent and one observation is gathered (along with one reward). The belief state on which the agent is based to build the Q function is made up of any length history of each element provided in the observation.

The code is based on Python (compatible with python 2.7 or python 3).  The framework is made in such a way that it is very easy to use your favorite python-based framework to code your own neural network architecture. The provided neural network architectures uses Theano (with or without the lasagne library).

## Dependencies

This framework is tested to work under Python 2.7, and Python 3.5. It should also work with Python 3.3 and 3.4.

The required dependencies are NumPy >= 1.10, joblib >= 0.9. You also need theano >= 0.7 (lasagne is optional) or you can write your own neural network using your favorite framework.

For running the examples, Matplotlib >= 1.1.1 is required. 
For running the atari games environment, you need to install ALE >= 0.4.
