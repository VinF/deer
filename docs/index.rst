Welcome to DeeR's documentation!
==================================

DeeR (Deep Reinforcement) is a python library to train an agent how to behave in a given environment so as to maximize a cumulative sum of rewards (see :ref:`what-is-deer`).

Here are key advantages of the library:

* You have access within a single library to techniques such as Double Q-learning, prioritized Experience Replay, Deep deterministic policy gradient (DDPG), etc.
* This package provides a general framework where observations are made up of any number of elements : scalars, vectors and frames. The belief state on which the agent is based to build the Q function or the policy is made up of any length history of each element provided in the observation.
* You can easily add up a validation phase that allows to stop the training process before overfitting. This possibility is useful when the environment is dependent on scarce data (e.g. limited time series).

In addition, the framework is made in such a way that it is easy to 

* build any environment
* modify any part of the learning process
* use your favorite python-based framework to code your own neural network architecture. The provided neural network architectures are based on Keras (or pure Theano) but you may easily use others.

It is a work in progress and input is welcome. Please submit any contribution via pull request.

What is new
------------
Version 0.3 
************
- Integration of different exploration/exploitation policies and possibility to easily built your own (see :ref:`policies`)
- Integration of DDPG for continuous action spaces (see :ref:`actor-critic`)
- :ref:`naming_conv` and some interfaces have been updated. This may cause broken backward compatibility. In that case, make the changes to the new convention by looking at the API in this documentation or by looking at the current version of the examples.
- Additional automated tests


Version 0.2
***********
- Standalone python package (you can simply do ``pip install deer``)
- Integration of new examples environments : :ref:`toy_env_pendulum`, :ref:`PLE` and :ref:`gym`
- Double Q-learning and prioritized Experience Replay
- Augmented documentation
- First automated tests

Future extensions:
******************

* Add planning (e.g. MCTS based when deterministic environment)
* Several agents interacting in the same environment
* ...


User Guide
------------

.. toctree::
  :maxdepth: 2
  
  user/installation
  user/tutorial
  user/environments
  user/development

API reference
-------------

If you are looking for information on a specific function, class or method, this API is for you.

.. toctree::
  :maxdepth: 2
  
  modules/agents
  modules/controllers
  modules/environments
  modules/q-networks
  modules/policies
  
Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/VinF/Deer
