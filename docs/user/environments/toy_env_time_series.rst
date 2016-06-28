.. _toy_env_time_series:

:mod:`Toy environment with time series`
=======================================

Description of the environement
###############################

This environment simulates the possibility of buying or selling a good. The agent can either have one unit or zero unit of that good. At each transaction with the market, the agent obtains a reward equivalent to the price of the good when selling it and the opposite when buying. In addition, a penalty of 0.5 (negative reward) is added for each transaction.

The price pattern is made by repeating the following signal plus a random constant between 0 and 3:

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example_signal.png
   :width: 250 px
   :alt: Toy example price pattern
   :align: center

Let's now see how this environement is built by looking into the file ``Toy_env.py`` in |toy_env_folder|. It is important to note that any environment derives from the base class :ref:`env_interface` and you can refer to it in order to understand the required methods and their usage.

.. |toy_env_folder| raw:: html

   <a href="https://github.com/VinF/deer/tree/master/examples/toy_env" target="_blank">examples/toy_env/</a>

..
    The price signal is built following the same rules for the training and the validation environments which allows the agent to learn a strategy that exploits this successfully.
    
    
    .. literalinclude:: ../../../examples/toy_env/Toy_env.py
       :language: python
       :lines: 21-75
    
    .. literalinclude:: ../../../examples/toy_env/Toy_env.py
       :language: python
       :lines: 116-130


How to run
##########

A minimalist way of running this example can be found in the file ``run_toy_env_simple.py`` in |toy_env_folder|.

* First, we need to import the agent, the Q-network, the environement and some controllers

.. literalinclude:: ../../../examples/toy_env/run_toy_env_simple.py
   :language: python
   :lines: 6-11
   :linenos:


* Then we instantiate the different elements as follows:

.. literalinclude:: ../../../examples/toy_env/run_toy_env_simple.py
   :language: python
   :lines: 13-51
   :linenos:


Results
########

Navigate to the folder ``examples/toy_env/`` in a terminal window. The example can then be run by using

.. code-block:: bash

    python run_toy_env_simple.py

You can also choose the full version of the launcher that specifies the hyperparameters for better performance.

.. code-block:: bash

    python run_toy_env.py

Every 10 epochs, a graph is saved in the 'toy_env' folder. You can then see that kind of behaviour for the test policy at the end of the training:

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/plot_toy_example.png
   :width: 250 px
   :alt: Toy example policy
   :align: center


In this graph, you can see that the agent has successfully learned to take advantage of the price pattern to buy when it is low and to sell when it is high. This example is of course easy due to the fact that the patterns are very systematic which allows the agent to successfuly learn it. It is important to note that the results shown are made on a validation set that is different from the training and we can see that learning generalizes well. For instance, the action of buying at time step 7 and 16 is the expected result because in average this will allow to make profit since the agent has no information on the future.

Using Convolutions VS LSTM's
############################

So far, the neural network was build by using a convolutional architecture as follows:

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/Convolutions_architecture.png
   :width: 350 px
   :alt: Convolutions architecture
   :align: center

The neural nework processes time series thanks to a set of convolutions layers. The output of the convolutions as well as the other inputs are followed by fully connected layers and the ouput layer.


When working with deep reinforcement learning, it is also possible to work with LSTM's (see for instance this |introduction-LSTM|)

.. |introduction-LSTM| raw:: html

   <a href="http://colah.github.io/posts/2015-08-Understanding-LSTMs/" target="_blank">introduction to LSTM's</a>

If you want to use LSTM's architecture, you can import the following libraries

.. code-block:: bash

    from deer.q_networks.q_net_keras import MyQNetwork
    from deer.q_networks.NN_keras_LSTM import NN as NN_keras

and then instanciate the qnetwork by specifying the 'neural_network' as follows:

.. code-block:: bash

     qnetwork = MyQNetwork(
         env,
         neural_network=NN_keras)
