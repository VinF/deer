Agents
==================

The NeuralAgent class wraps up all methods for exploring, training and testing in any given environment.

It relies on controllers to modify its hyper parameters through time and to decide how the validation/test procedures should happen. Controllers are attached to an agent using the agent.attach(Controller) method. 

Controllers
------------

All controllers should inherit from the base Controller class (which does nothing when receiving the various signals emitted by an agent). The following methods are defined in this base controller class:

* `__init__(self)`: Activate the controller.
All controllers inheriting this class should call this method in their own `__init()__` using `super(self.__class__, self).__init__()`.
* `setActive(self, active)`: Activate or deactivate this controller. A controller should not react to any signal it receives as long as it is deactivated. For instance, if a controller maintains a counter on how many episodes it has seen, this counter should not be updated when this controller is disabled.
* `OnStart(self, agent)`: Called when the agent is going to start working (before anything else). This corresponds to the moment where the agent's `run()` method is called.
* `OnEpisodeEnd(self, agent, terminalReached, reward)`: Called whenever the agent ends an episode, just after this episode ended and before any `OnEpochEnd()` signal could be sent.