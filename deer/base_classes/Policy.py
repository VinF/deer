class Policy(object):

    def __init__(self, environment_):
        self.environment = environment_

    def act(self, state):
        raise NotImplementedError()