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
   wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
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
      numpy=$NUMPY_VERSION scipy=$SCIPY_VERSION \
      matplotlib theano=$THEANO_VERSION joblib

source activate testenv


if [[ "$COVERAGE" == "true" ]]; then
    pip install coverage coveralls
fi

python --version
python -c "import numpy; print('numpy %s' % numpy.__version__)"
python -c "import scipy; print('scipy %s' % scipy.__version__)"
python -c "import theano; print('theano %s' % theano.__version__)"

python setup.py develop
