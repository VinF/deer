class Policy(object):
    """Abstract class for all policies, i.e. objects that can take any space as input, and output an action.
    """

    def __init__(self, environment_, dataset_=None):
        self.environment = environment_
        self.dataset = dataset_

    def act(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()

    def train(self):
        """If a dataset is used by the policy, it can be used for training.
        """
        pass

    def update_after_action(self):
        pass

    def update_after_episode(self):
        pass

    def update_after_epoch(self):
        pass