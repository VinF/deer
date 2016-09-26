.. _two_storages:

:mod:`Two storage devices environment`
========================================

Description of the environement
###############################

This example simulates the operation of a realistic micro-grid (such as a smart home for instance) that is not connected to the main utility grid (off-grid) and that is provided with PV panels, batteries and hydrogen storage. The battery has the advantage that it is not limited in instaneous power that it can provide or store. The hydrogen storage has the advantage that is can store very large quantity of energy.

.. code-block:: bash

    python run_MG_two_storage_devices


This example uses the environment defined in MG_two_storage_devices_env.py. The agent can either choose to store in the long term storage or take energy out of it while the short term storage handle at best the lack or surplus of energy by discharging itself or charging itself respectively. Whenever the short term storage is empty and cannot handle the net demand a penalty (negative reward) is obtained equal to the value of loss load set to 2euro/kWh.

The state of the agent is made up of an history of two to four punctual observations:

* Charging state of the short term storage (0 is empty, 1 is full)
* Production and consumption (0 is no production or consumption, 1 is maximal production or consumption)
* (Distance to equinox)
* (Predictions of future production : average of the production for the next 24 hours and 48 hours)

Two actions are possible for the agent:

* Action 0 corresponds to discharging the long-term storage
* Action 1 corresponds to charging the long-term storage

More information can be found in the paper to be published :
    Deep Reinforcement Learning Solutions for Energy Microgrids Management, Vincent François-Lavet, David Taralla, Damien Ernst, Raphael Fonteneau


Annex to the paper
##################

..
    Neural network architecture
    ***************************
    
    We propose a neural network architecture where the inputs are provided by the state vector, and where each separate output represents the Q-value function for one of the discretized actions. The action :math:`a_t` to be made at time :math:`t` is whether to charge or discharge the hydrogen storage device with the assumption that the batteries handle at best the current demand (avoid any value of loss load whenever possible). We consider three discretized actions : (i) discharge at full rate the hydrogen storage, (ii) keep it idle or (iii) charge it at full rate.
    
    The neural network process time series thanks to a set of convolutions that convolves 16 filters of :math:`2 \times 1` with stride 1 followed by a convolution with 16 filters of :math:`2 \times 2` with stride 1. The output of the convolutions as well as the other inputs are then followed by two fully connected layers with 50 and 20 neurons and the ouput layer. The activation function used is the Rectified Linear Unit (ReLU) except for the output layer where no activation function is used. 
    
    .. figure:: http://vincent.francois-l.be/img_GeneralDeepQRL/Convolutions_architecture.png
       :width: 400 px
       :align: center
    
       Sketch of the structure of the neural network architecture (without representing the actual number of neurons in each layer). The neural network processes time series thanks to a set of convolutions layers. The output of the convolutions as well as the other inputs are followed by fully connected layers and the ouput layer.


PV production and consumption profiles
**************************************
Solar irradiance varies throughout the year depending on the seasons, and it also varies throughout the day depending on the weather and the position of the sun in the sky relative to the PV panels. The main distinction between these profiles is the difference between summer and winter PV production. In particular, production varies with a factor 1:5 between winter and summer as can be seen from the measurements of PV panels production for a residential customer located in Belgium in the figures below. 

.. figure:: http://vincent.francois-l.be/img_GeneralDeepQRL/ProductionVSMonths_be.png
   :width: 300 px
   :align: center
   
   Total energy produced per month

.. figure:: http://vincent.francois-l.be/img_GeneralDeepQRL/ProductionVSTime_1janv_be.png
   :width: 300 px
   :align: center
   
   Typical production in winter

.. figure:: http://vincent.francois-l.be/img_GeneralDeepQRL/ProductionVSTime_1july_be.png
   :width: 300 px
   :align: center

   Typical production in summer
   

A simple residential consumption profile is considered with a daily average consumption of 18kWh (see figure below). 

.. figure:: http://vincent.francois-l.be/img_GeneralDeepQRL/ConsumptionVSTime_random.png
   :width: 300 px
   :align: center

   Representative residential consumption profile



Main microgrid parameters
**************************

.. list-table:: Data used for the PV panels
   :widths: 30 10 20

   * - cost
     - :math:`c^{PV}`
     - :math:`1 euro/W_p`
   * - Efficiency
     - :math:`\eta^{PV}`
     - :math:`18 \%`
   * - Life time
     - :math:`L^{PV}`
     - :math:`20 years`

.. list-table:: Data used for the :math:`LiFePO_4` battery
   :widths: 30 10 20

   * - cost
     - :math:`c^B`
     - :math:`500 euro/kWh`
   * - discharge efficiency
     - :math:`\eta_0^B`
     - :math:`90\%`
   * - charge efficiency
     - :math:`\zeta_0^B`
     - :math:`90\%`
   * - Maximum instantaneous power
     - :math:`P^B`
     - :math:`> 10kW`
   * - Life time
     - :math:`L^{B}`
     - :math:`20 years`

.. list-table:: Data used for the Hydrogen storage device
   :widths: 30 10 20

   * - cost
     - :math:`c^{H_2}`
     - :math:`14 euro/W_p`
   * - discharge efficiency
     - :math:`\eta_0^{H_2}`
     - :math:`65\%`
   * - charge efficiency
     - :math:`\zeta_0^{H_2}`
     - :math:`65\%`
   * - Life time
     - :math:`L^{H_2}`
     - :math:`20 years`
     
.. list-table:: Data used for reward function
   :widths: 30 10 20

   * - cost endured per kWh not supplied within the microgrid
     - :math:`k`
     - :math:`2 euro/kWh`
   * - revenue/cost per kWh of hydrogen produced/used
     - :math:`k^{H_2}` 
     - :math:`0.1 euro/kWh`
