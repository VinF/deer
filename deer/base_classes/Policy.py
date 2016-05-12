class Policy(object):
    """Abstract class for all policies, i.e. objects that can take any space as input, and output an action.
    """

    def __init__(self, q_network, n_actions,random_state):
        self.q_network = q_network
        self.n_actions = n_actions
        self.random_state = random_state

        pass

    def bestAction(self, state):
        """ Returns the best Action
        """
        action = self.q_network.chooseBestAction(state)
        V = max(self.q_network.qValues(state))
        return action, V

    def act(self, state):
        """Main method of the Policy class. It can be called by agent.py, given a state,
        and should return a valid action w.r.t. the environment given to the constructor.
        """
        raise NotImplementedError()
