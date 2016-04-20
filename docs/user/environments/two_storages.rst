.. _two_storages:

:mod:`Two storage devices environment`
========================================

This example simulates the operation of a realistic micro-grid (such as a smart home for instance) that is not connected to the main utility grid (off-grid) and that is provided with PV panels, batteries and hydrogen storage. The battery has the advantage that it is not limited in instaneous power that it can provide or store. The hydrogen storage has the advantage that is can store very large quantity of energy.

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
