Environments
==================

The environment defines the dynamics and the reward signal that the agent observes when interacting with it.
    
An agent sees at any time-step from the environment a collection of observable elements. Observing the environment at time t thus corresponds to obtaining a punctual observation for each of these elements. According to the control problem to solve, it might be useful for the agent to not only take action based on the current punctual observation but rather on a collection of the last punctual observations. In this framework, it's the environment that defines the number of each punctual observation to be considered.

Different "modes" are used in this framework to allow the environment to have different dynamics and/or reward signal. For instance, in training mode, only a part of the dynamics may be available so that it is possible to see how well the agent generalizes to a slightly different one.

All the Environment that the user wants to use should inherit the Environment interface that is provided in base_classes.py and defines the following methods:
    
* `reset(self, mode)`: Reset the environment and put it in mode *mode*.
* `act(self, action, mode)`: Apply the agent action `action `on the environment.
* `inputDimensions(self)`: Get the shape of the input space for this environment.
* `nActions(self)`: Get the number of different actions that can be taken on this environment.
* `inTerminalState(self)`: Tell whether the environment reached a terminal state after the last transition (i.e. the last transition that occured was terminal).
* `observe(self)`: Get a punctual observation for each of the elements composing this environment.


.. toctree::
  :maxdepth: 2
  
  environments/toy_env_pendulum.rst
  environments/toy_env_time_series.rst
  environments/two_storages.rst  
  environments/ALE.rst