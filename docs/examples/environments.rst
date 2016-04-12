Environments
==================

The environment defines the dynamics and the reward signal that the agent observes when interacting with it.
    
An agent sees at any time-step from the environment a collection of observable elements. Observing the environment at time t thus corresponds to obtaining a punctual observation for each of these elements. According to the control problem to solve, it might be useful for the agent to not only take action based on the current punctual observation but rather on a collection of the last punctual observations. In this framework, it's the environment that defines the number of each punctual observation to be considered.

Different "modes" are used in this framework to allow the environment to have different dynamics and/or reward signal. For instance, in training mode, only a part of the dynamics may be available so that it is possible to see how well the agent generalizes to a slightly different one.

All the Environment that the user wants to use should inherit the Environment interface that is provided in base_classes.py and defines the following methods:
    
* `reset(self, mode)`: Reset the environment and put it in mode `mode`.
* `act(self, action, mode)`: Apply the agent action `action `on the environment.
* `inputDimensions(self)`: Get the shape of the input space for this environment.
* `nActions(self)`: Get the number of different actions that can be taken on this environment.
* `inTerminalState(self)`: Tell whether the environment reached a terminal state after the last transition (i.e. the last transition that occured was terminal).
* `observe(self)`: Get a punctual observation for each of the elements composing this environment.


Toy environment with time series
--------------------------------

The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction. 

This example uses the environment defined in Toy_env.py.
The state of the agent is made up of a past history of two punctual observations:
* The price signal
* Either the agent possesses the good or not (1 or 0)
It is important to note that the agent has no direct information about the future price signal.

Two actions are possible for the agent:
* Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
* Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.


The price pattern is made by repeating the following signal plus a random constant between 0 and 3:
<div align="center">
<img src="http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example_signal.png" alt="Toy example"  width="250" />
</div> 
The price signal is built following the same rules for the training and the validation environments which allows the agent to learn a strategy that exploits this successfully.

### Results
This example can be run by using 
.. code-block:: bash
    
python run_toy_env

After ten epochs, the following graph is obtained:
<div align="center">
<img src="http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example.png" width="250" alt="Toy example">
</div>

In this graph, you can see that the agent has successfully learned after 10 epochs to take advantage of the price pattern to buy when it is low and to sell when it is high. This example is of course easy due to the fact that the patterns are very systematic which allows the agent to successfuly learn it. It is important to note that the results shown are made on a validation set that is different from the training and we can see that learning generalizes well. For instance, the action of buying at time step 7 and 16 is the expected result because in average this will allow to make profit since the agent has no information on the future.


Two storage devices environment
---------------------------------

This second example is slightly more complex and realistic. It simulates the operation of a micro-grid (such as a smart home for instance) that is not connected to the main utility grid (off-grid) and that is provided with PV panels, batteries and hydrogen storage. The battery has the advantage that it is not limited in instaneous power that it can provide or store. The hydrogen storage has the advantage that is can store very large quantity of energy.

.. code-block:: bash
    
python run_MG_two_storage_devices

This example uses the environment defined in MG_two_storage_devices_env.py. The agent can either choose to store in the long term storage or take energy out of it while the short term storage handle at best the lack or surplus of energy by discharging itself or charging itself respectively. Whenever the short term storage is empty and cannot handle the net demand a penalty (negative reward) is obtained equal to the value of loss load set to 2euro/kWh.

The state of the agent is made up of an history of two to four punctual observations:

* Charging state of the short term storage (0 is empty, 1 is full)
* Production and consumption (0 is no production or consumption, 1 is maximal production or consumption)
* ( Distance to equinox )
* ( Predictions of future production : average of the production for the next 24 hours and 48 hours )

Two actions are possible for the agent:

* Action 0 corresponds to discharging the long-term storage
* Action 1 corresponds to charging the long-term storage

More information can be found in the paper to be published :
    Efficient decision making in stochastic micro-grids using deep reinforcement learning, Vincent François-Lavet, David Taralla, Damien Ernst, Raphael Fonteneau


ALE environment
------------------

This environment is an interface with the ALE environment that simulates any ATARI game. The hyper-parameters aim to simulate as closely as possible the following paper : Mnih, Volodymyr, et al. "Human-level control through deep reinforcement learning." Nature 518.7540 (2015): 529-533.

Some changes are still necessary to obtain the same performances.

