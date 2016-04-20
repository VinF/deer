Tutorial
=========

.. _what-is-deer:

What is deep reinforcement learning?
------------------------------------

Deep reinforcement learning is the combination of two fields:

* *Reinforcement learning (RL)* is a theory that allows an agent to learn a startegy so as to maximize a sum of cumulated (delayed) rewards from any given environement. If you are not familiar with RL, you can get up to speed easily with |SB_link| by Sutton and Barto.

.. |SB_link| raw:: html

   <a href="https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html" target="_blank">this book</a>
   
   
* *Deep learning* is a branch of machine learning for regression and classification. It is particularly well suited to model high-level abstractions in data by using multiple processing layers composed of multiple non-linear transformations.

This combination allows to learn complex tasks such as playing ATARI games from high-dimensional sensory inputs. For more informations, you can refer to one of the main paper in the domain : |Human-level_link|.

.. |Human-level_link| raw:: html

   <a href="http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html" target="_blank">"Human-level control through deep reinforcement learning"</a>

..
    How does it work?
    -------------------

    In RL, there are two main parts:

    * An agent with learning capabilities.
    * An environment. 

    The environment defines the task to be performed by the agent with the following elements:

    * a set of environment states S
    * a set of actions A
    * a dynamics of the system, i.e. rules of transitioning between states
    * a reward function, i.e rules that determine the immediate reward (scalar) of a transition
    * a set of obsevrations O, that may be the same than S (MDP case) or different (POMDP case)


How can I get started?
-----------------------

First, make sure you have installed the package properly by following the steps descibed in :ref:`installation`.

The general idea of this framework is that you need to instantiate an agent (along with a q-network) and an environment. In order to perform an experiment, you also need to attach to the agent some controllers for controlling the training and the various parameters of your agent.

The environment should be built specifically for any specific task while q-networks, the DQN agent and many controllers are provided within this package. 

The best to get started is to have a look at the :ref:`examples` and in particular the two first environments that are simple to understand: 

* :ref:`toy_env_time_series`
* :ref:`toy_env_pendulum`

If you find something that is not yet implemented and if you wish to contribute, you can check the section :ref:`dev`.

..
    From there, you can look at this documentation for more informations on the controllers and the other environments. 


