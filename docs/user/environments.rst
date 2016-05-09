.. _examples:

Examples
========

You can find these examples at the |package_root|. For each example at least two files are provided:

* A launcher file (whose name usually starts by ``run_``). 
* An environnement file (whose name usually ends by ``_env``). 


.. |package_root| raw:: html

   <a href="https://github.com/VinF/deer/tree/master/examples" target="_blank">root of the package</a>


The launcher file performs different actions:

* It instantiates the environment, the agent (along with a q-network).
* It binds controllers to the agent
* it finally runs the experiment

Examples are better than precepts and the best is to get started with the following examples (with the simplest examples listed first)

.. toctree::
  :maxdepth: 2
  
  environments/toy_env_time_series.rst
  environments/toy_env_pendulum.rst
  environments/gym.rst
  environments/two_storages.rst  
  environments/ALE.rst
  environments/PLE.rst