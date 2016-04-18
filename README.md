## Full Documentation

See the [Wiki](https://github.com/VinF/deer/wiki) for full documentation, examples and other information.

## Dependencies

This framework is tested to work under Python 2.7, and Python 3.5. It should also work with Python 3.3 and 3.4.

The required dependencies are NumPy >= 1.10, joblib >= 0.9. You also need theano >= 0.7 (lasagne is optional) or you can write your own neural network using your favorite framework.

For running the examples, Matplotlib >= 1.1.1 is required. 
For running the atari games environment, you need to install ALE >= 0.4.

## How to install
You can simply clone the version 0.1 of this framework by using the following command:
```
git clone -b 0.1 https://github.com/VinF/deer.git
```
That version is not a package yet, so you can simply launch it as a standalone python code.


For the latest developments, you can instead clone the bleeding-edge version of this framework by using the following command:
```
git clone -b master https://github.com/VinF/deer.git
```
That version is not a package yet, so you can simply launch it as a standalone python code.


For the latest developments, you can instead clone the bleeding-edge version of this framework by using the following command:
```
git clone -b master https://github.com/VinF/General_Deep_Q_RL.git
```

Assuming you already have a python environment with pip, you can automatically install all the dependencies (except ALE that you may need for atari games) with:
```
pip install -r requirements.txt
```

And you can install the framework as a package with:
```
python setup.py install
```
