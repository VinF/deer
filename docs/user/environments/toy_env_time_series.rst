.. _toy_env_time_series:

:mod:`Toy environment with time series`
=======================================

Description
########### 

The environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction. 

This example is defined in ``examples/toy_env``.
The state of the agent is made up of a past history of two punctual observations:

* The price signal
* Either the agent possesses the good or not (1 or 0)

It is important to note that the agent has no direct information about the future price signal.

Two actions are possible for the agent:

* Action 0 corresponds to selling if the agent possesses one unit or idle if the agent possesses zero unit.
* Action 1 corresponds to buying if the agent possesses zero unit or idle if the agent already possesses one unit.


The price pattern is made by repeating the following signal plus a random constant between 0 and 3:

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example_signal.png
   :width: 250 px
   :alt: Toy example price pattern
   :align: center


The price signal is built following the same rules for the training and the validation environments which allows the agent to learn a strategy that exploits this successfully.

Results
########

Navigate to the folder ``examples/toy_env`` in a terminal window. The example can then be run by using

.. code-block:: bash

    python run_toy_env.py

After 10 epochs, the following graph is obtained:

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example.png
   :width: 250 px
   :alt: Toy example policy
   :align: center


In this graph, you can see that the agent has already successfully learned to take advantage of the price pattern to buy when it is low and to sell when it is high. This example is of course easy due to the fact that the patterns are very systematic which allows the agent to successfuly learn it. It is important to note that the results shown are made on a validation set that is different from the training and we can see that learning generalizes well. For instance, the action of buying at time step 7 and 16 is the expected result because in average this will allow to make profit since the agent has no information on the future.
