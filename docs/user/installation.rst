.. _installation:

Installation
==============


Dependencies
--------------

This framework is tested to work under Python 3.6.

The required dependencies are NumPy >= 1.10, joblib >= 0.9. You also need keras or you can write your own learning algorithms using your favorite deep learning framework.

For running some of the examples, Matplotlib >= 1.1.1 is required. You also sometimes need to install specific dependencies (e.g. for the atari games, you need to install ALE >= 0.4).


We recommend to use the bleeding-edge version and to install it by following the :ref:`dev-install`. If you want a simpler installation procedure and do not intend to modify yourself the learning algorithms etc., you can look at the :ref:`user-install`. 

.. _dev-install:

Developer install instructions
-------------------------------

As a developer, you can set you up with the bleeding-edge version of DeeR with: 

.. code-block:: bash

    git clone -b master https://github.com/VinF/deer.git

Assuming you already have a python environment with ``pip``, you can automatically install all the dependencies (except specific dependencies that you may need for some examples) with:

.. code-block:: bash
    
    pip install -r requirements.txt


And you can install the framework as a package using the mode ``develop`` so that you can make modifications and test without having to re-install the package.

.. code-block:: bash
    
    python setup.py develop


.. _user-install:

User install instructions
--------------------------

You can install the framework with pip:

.. code-block:: bash
    
    pip install deer

For the bleeding edge version (recommended), you can simply use

.. code-block:: bash

    pip install git+git://github.com/VINF/deer.git@master

    
..
    If you want to update it to the bleeding edge version you can use pip for this with the command line below:
 
    .. code-block:: bash
    
        pip install --upgrade --no-deps git+git://github.com/VinF/deer


