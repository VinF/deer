.. -*- mode: rst -*-

|Travis|_ |Python27|_ |Python35|_ |PyPi|_ |License|_

.. |Travis| image:: https://travis-ci.org/VinF/deer.svg?branch=master
.. _Travis: https://travis-ci.org/VinF/deer

.. |Python27| image:: https://img.shields.io/badge/python-2.7-blue.svg
.. _Python27: https://badge.fury.io/py/deer

.. |Python35| image:: https://img.shields.io/badge/python-3.5-blue.svg
.. _Python35: https://badge.fury.io/py/deer

.. |PyPi| image:: https://badge.fury.io/py/deer.svg
.. _PyPi: https://badge.fury.io/py/deer

.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg
.. _License: https://github.com/VinF/deer/blob/master/LICENSE

DeeR
====

DeeR is a python library for Deep Reinforcement. It is build with modularity in mind so that it can easily be adapted to any need. It provides many possibilities out of the box (prioritized experience replay, double Q-learning, etc). Many different environment examples are also provided (some of them using OpenAI gym). 

Full Documentation
==================

The documentation is available at : http://deer.readthedocs.io/en/master/

Dependencies
============

This framework is tested to work under Python 2.7, and Python 3.5. It should also work with Python 3.3 and 3.4.

The required dependencies are NumPy >= 1.10, joblib >= 0.9. You also need theano >= 0.7 (lasagne is optional) or you can write your own neural network using your favorite framework.

For running the examples, Matplotlib >= 1.1.1 is required. 
For running the atari games environment, you need to install ALE >= 0.4.