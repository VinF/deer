"""This module contains classes used to define an agent suited for playing with the ALE environment.

See environments.ALE_env, run_ALE.

Authors: Vincent Francois-Lavet, David Taralla
"""

from .agent import NeuralAgent

class ALEAgent(NeuralAgent):
    def _chooseAction(self):
        if self._mode != -1:
            if self._randomState.rand() < 0.05:
                action = self._random_state.randint(0, self._environment.nActions())
                V = 0
            else:
                action, V = self.bestAction()
        else:
            if self._dataset.n_elems > self._replay_start_size:
                # e-Greedy policy
                if self._random_state.rand() < self._epsilon:
                    action = self._random_state.randint(0, self._environment.nActions())
                    V = 0
                else:
                    action, V = self.bestAction()
            else:
                # Still gathering initial data: choose dummy action
                action = self._random_state.randint(0, self._environment.nActions())
                V = 0
                
        for c in self._controllers: c.onActionChosen(self, action)
        return action, V

if __name__ == "__main__":
    pass
