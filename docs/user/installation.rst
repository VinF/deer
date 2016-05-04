.. _installation:

Installation
==============


Dependencies
--------------

This framework is tested to work under Python 2.7, and Python 3.5. It should also work with Python 3.3 and 3.4.

The required dependencies are NumPy >= 1.10, joblib >= 0.9. You also need theano >= 0.7 (lasagne is optional) or you can write your own neural network using your favorite framework.

For running some of the examples, Matplotlib >= 1.1.1 is required. You also sometimes need to install specific dependencies (e.g. for the atari games, you need to install ALE >= 0.4).


User install instructions
--------------------------

The easiest is to install the framework with pip:

.. code-block:: bash
    
    pip install deer

    
..
    If you want to update it to the bleeding edge version you can use pip for this with the command line below:
 
    .. code-block:: bash
    
        pip install --upgrade --no-deps git+git://github.com/VinF/deer


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


