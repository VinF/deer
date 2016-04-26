.. _`agents`:

:mod:`Agents`
=============

The NeuralAgent class is at the core of the framework for exploring, training and testing in a given environment. 

It relies on :ref:`controllers` to modify its hyper parameters through time and to decide how the validation/test procedures should happen. Controllers are attached to an agent using the ``agent.attach(Controller)`` method. 

.. autoclass:: ..deer.agent.NeuralAgent
