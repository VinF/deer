.. _toy_env_pendulum:

:mod:`The pendulum on a cart`
=============================


Description
########### 

The environment simulates the behavior of an inverted pendulum. The theoretical system with its equations are as described in |barto-sutton-anderson|:

.. |barto-sutton-anderson| raw:: html

   <a href="https://webdocs.cs.ualberta.ca/~sutton/papers/barto-sutton-anderson-83.pdf" target="_blank">Barto et al. (1983)</a>


* A cart of mass :math:`M` that can move horizontally;
* A pole of mass :math:`m` and length :math:`l` attached to the cart, with :math:`\theta` in :math:`[0, -\pi]` for the lefthand plane, and :math:`[0, \pi]` for the righthand side. We are supposing that the cart is moving on a rail and the pole can go under it.

.. image:: https://upload.wikimedia.org/wikipedia/commons/thumb/0/00/Cart-pendulum.svg/2000px-Cart-pendulum.svg.png
   :width: 200 px
   :alt: Inverted Pendulum
   :align: center


The goal of the agent is to balance the pole above its supporting cart (:math:`\theta=0`), by displacing the cart left or right - thus, 2 actions are possible. To do so, the environment communicates to the agent:

* A vector (position, speed, angle, angular speed);
* The reward associated to the action chosen by the agent. 

Results
########

In a terminal windown go to the folder ``examples/pendulum``. The example can then be run with 

.. code-block:: bash

    python run_pendulum.py

Here are the outputs of the agent after respectively 20 and 70 learning epochs, with 1000 steps in each. We clearly see the final success of the agent in controlling the inverted pendulum. 

Note: a MP4 is generated every `PERIOD_BTW_SUMMARY_PERFS` epochs and you need the [FFmpeg](https://www.ffmpeg.org/) library to do so. If you do not want to install this library or to generate the videos, just set `PERIOD_BTW_SUMMARY_PERFS = -1`.

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/output2.gif
   :width: 500 px
   :align: center

.. image:: http://vincent.francois-l.be/img_GeneralDeepQRL/output7.gif
   :width: 500 px
   :align: center

Details on the implementation
##############################

The main focus in the environment is to implement `act(self, action)` which specifies how the cart-pole system behaves in response to an input action. So first, we transcript the physical laws that rule the motion of the pole and the cart. The simulation timestep of the agent is :math:`\Delta_t=0.02` second. But we discretize this value even further in `act(self, action)`, in order to obtain dynamics that are closer to the exact differential equations. 
Secondly, we chose the reward function as the sum of :

* :math:`- |\theta|` such that the agent receives 0 when the pole is standing up, and a negative reward proportional to the angle otherwise.
* :math:`- \frac{|x|}{2}` such that the agent receives a negative reward when it is far from :math:`x=0`.