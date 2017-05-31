# inspired from scikit-learn contrib

set -e

if [[ "$EXAMPLE" == "toy_env" ]]; then
    cd examples/toy_env
    python run_toy_env.py --epochs 5
    python run_toy_env_simple.py & sleep 30; kill $!

elif [[ "$EXAMPLE" == "mountain_car" ]]; then
    pip install gym
    cd examples/gym
    python run_mountain_car.py  --epochs 5

    pip -V pip
    pip install --upgrade pip
    conda install libgcc -y
    if [[ "$PYTHON_VERSION" == "2.7" ]]; then
      pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp27-none-linux_x86_64.whl;
    elif [[ "$PYTHON_VERSION" == "3.5" ]]; then
      pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.10.0-cp35-cp35m-linux_x86_64.whl;
    fi
    
#    pip install keras
#    python run_mountain_car_continuous.py  --epochs 5

else
    # Get into a temp directory to run test from the installed and
    # check if we do not leave artifacts
    mkdir -p $TEST_DIR

    cd $TEST_DIR

    if [[ "$COVERAGE" == "true" ]]; then
        nosetests -vs --with-coverage --cover-package=$MODULE $MODULE
    else
        nosetests -vs $MODULE
    fi

fi
