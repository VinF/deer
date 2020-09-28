# inspired from scikit-learn contrib

# Deactivate the travis-provided virtual environment and setup a
# conda-based environment instead
deactivate

# Use the miniconda installer for faster download / install of conda
# itself
pushd .
cd
mkdir -p download
cd download
echo "Cached in $HOME/download :"
ls -l
echo
if [[ ! -f miniconda.sh ]]
   then
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \ #Miniconda3-4.5.4-Linux-x86_64.sh \
       -O miniconda.sh
   fi
chmod +x miniconda.sh && ./miniconda.sh -b
cd ..
ls /home/travis
export PATH=/home/travis/miniconda/bin:$PATH
export PATH=/home/travis/miniconda2/bin:$PATH
export PATH=/home/travis/miniconda3/bin:$PATH
conda update --yes conda
popd

# Configure the conda environment and put it in the path using the
# provided versions
conda create -n testenv --yes python=$PYTHON_VERSION pip nose \
      numpy

conda install libgcc -y
source activate testenv
pip install --upgrade pip
pip install scipy
pip install tensorflow
pip install keras
pip install matplotlib
pip install joblib
pip install cython

#if [[ "$PYTHON_VERSION" == "2.7" ]]; then
#    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp27-none-linux_x86_64.whl # tensorflow
#elif [[ "$PYTHON_VERSION" == "3.5" ]]; then
#    pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-1.10.1-cp35-cp35m-linux_x86_64.whl 
#fi

if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import theano; print('theano %s' % theano.__version__)"
python -c "import tensorflow; print('tensorflow %s' % tensorflow.__version__)"

python setup.py develop
