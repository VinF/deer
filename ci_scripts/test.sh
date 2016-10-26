# inspired from scikit-learn contrib

set -e

if [[ "$EXAMPLE" == "toy_env" ]]; then
    cd examples/toy_env
    python run_toy_env.py --epochs 5

elif [[ "$EXAMPLE" == "toy_env" ]]; then
    cd examples/toy_env
    python run_toy_env_simple.py --epochs 5

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
