Tutorial
==================

.. _what-is-deeprl:

What is deep reinforcement learning?
------------------------------------

The basic principles rely on the theory of reinforcement learning (RL). Basically, it is a theory that allows an agent to learn a startegy so as to maximize a sum of cumulated (delayed) rewards from any given environement. If you are not familiar with RL, you can get up to speed easily with this `book by Sutton and Barto 
<https://webdocs.cs.ualberta.ca/~sutton/book/the-book.html>`_.


Deep RL is the combination of deep learning and RL to provide the possibility to learn from high-dimensional sensory inputs to perform complex taks such as playing ATARI games. This was published in the paper `"Human-level control through deep reinforcement learning"  
<http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html>`_.


How does it work in practice?
-----------------------------
Two important parts are 

1. The environnement
2. The agent with learning capabilities

First, the environment can be viewed as the task to be performed by the agent. The goal of the learning is to obtain a strong strategy of maximizing rewards. 

An environnement is defined with the following elements:

* a set of environment states S
* a set of actions A
* a dynamics of the system, i.e. rules of transitioning between states
* a reward function, i.e rules that determine the scalar immediate reward of a transition
* a set of obsevrations O, that may be the same than S (MDP case) or different (POMDP case)

An agent 


How can I run a first example?
-----------------------------


