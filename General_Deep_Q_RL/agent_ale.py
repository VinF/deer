"""
The NeuralAgent class wraps a deep Q-network for training and testing
in any given environment.

Modifications: Vincent Francois-Lavet
Contributor: David Taralla
"""

from agent import NeuralAgent

class ALEAgent(NeuralAgent):
    def _chooseAction(self):
        if self._mode != -1:
            if self._randomState.rand() < 0.05:
                action = self._randomState.randint(0, self._environment.nActions())
                V = 0
            else:
                action, V = self.bestAction()
        else:
            if self._dataSet.nElems() > self._replayMemoryStartSize:
                # e-Greedy policy
                if self._randomState.rand() < self._epsilon:
                    action = self._randomState.randint(0, self._environment.nActions())
                    V = 0
                else:
                    action, V = self.bestAction()
            else:
                # Still gathering initial data: choose dummy action
                action = self._randomState.randint(0, self._environment.nActions())
                V = 0
                
        for c in self._controllers: c.OnActionChosen(self, action)
        return action, V

if __name__ == "__main__":
    pass
