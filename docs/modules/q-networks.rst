:mod:`Learning algorithms`
==========================

Q-learning
---------------------
.. autosummary::

    deer.base_classes.QNetwork
    deer.q_networks.q_net_theano.MyQNetwork
    deer.q_networks.q_net_keras.MyQNetwork

.. _actor-critic:

Actor-critic learning
---------------------
.. autosummary::
    deer.q_networks.AC_net_keras.MyACNetwork

Detailed description
--------------------

.. autoclass:: deer.base_classes.QNetwork
   :members:
.. autoclass:: deer.q_networks.q_net_theano.MyQNetwork
   :members:
   :show-inheritance:
.. autoclass:: deer.q_networks.AC_net_keras.MyACNetwork
   :members:
   :show-inheritance:
.. autoclass:: deer.q_networks.q_net_keras.MyQNetwork
   :members:
   :show-inheritance:
