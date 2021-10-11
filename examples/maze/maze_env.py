""" Environment with a distribution of mazes (one new maze is drawn at each episode)

Author: Vincent Francois-Lavet
"""
# import matplotlib
# matplotlib.use('qt5agg')
# from mpl_toolkits.axes_grid1 import host_subplot
# import mpl_toolkits.axisartist as AA
# import matplotlib.pyplot as plt
import copy

import a_star_path_finding as pf
import numpy as np

from deer.base_classes import Environment


class MyEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self._random_state = rng
        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0
        self._episode_steps = 0
        self._actions = [0, 1, 2, 3]
        self._size_maze = 8
        self._higher_dim_obs = kwargs.get("higher_dim_obs", False)
        self._reverse = kwargs.get("reverse", False)

        self._n_walls = int(
            (self._size_maze - 2) ** 2 / 3.0
        )  # int((self._size_maze)**2/3.)
        self._n_rewards = 3
        self.create_map()
        self.intern_dim = 3

    def create_map(self):
        valid_map = False
        while valid_map == False:
            # Agent
            self._pos_agent = [1, 1]

            # Walls
            self._pos_walls = []
            for i in range(self._size_maze):
                self._pos_walls.append([i, 0])
                self._pos_walls.append([i, self._size_maze - 1])
            for j in range(self._size_maze - 2):
                self._pos_walls.append([0, j + 1])
                self._pos_walls.append([self._size_maze - 1, j + 1])

            n = 0
            while n < self._n_walls:
                potential_wall = [
                    self._random_state.randint(1, self._size_maze - 2),
                    self._random_state.randint(1, self._size_maze - 2),
                ]
                if (
                    potential_wall not in self._pos_walls
                    and potential_wall != self._pos_agent
                ):
                    self._pos_walls.append(potential_wall)
                    n += 1

            # Rewards
            # self._pos_rewards=[[self._size_maze-2,self._size_maze-2]]
            self._pos_rewards = []
            n = 0
            while n < self._n_rewards:
                potential_reward = [
                    self._random_state.randint(1, self._size_maze - 1),
                    self._random_state.randint(1, self._size_maze - 1),
                ]
                if (
                    potential_reward not in self._pos_rewards
                    and potential_reward not in self._pos_walls
                    and potential_reward != self._pos_agent
                ):
                    self._pos_rewards.append(potential_reward)
                    n += 1

            valid_map = self.is_valid_map(
                self._pos_agent, self._pos_walls, self._pos_rewards
            )

    def is_valid_map(self, pos_agent, pos_walls, pos_rewards):
        a = pf.AStar()
        pos_walls
        walls = [tuple(w) for w in pos_walls]
        start = tuple(pos_agent)
        for r in pos_rewards:
            end = tuple(r)
            a.init_grid(self._size_maze, self._size_maze, walls, start, end)
            maze = a
            optimal_path = maze.solve()
            if optimal_path == None:
                return False

        return True

    def reset(self, mode):
        self._episode_steps = 0
        self._mode = mode
        self.create_map()

        if mode == MyEnv.VALIDATION_MODE:
            if self._mode != MyEnv.VALIDATION_MODE:
                self._mode = MyEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0

            else:
                self._mode_episode_count += 1

        return [1 * [self._size_maze * [self._size_maze * [0]]]]

    def act(self, action):
        self._episode_steps += 1
        action = self._actions[action]

        reward = -0.1

        if action == 0:
            if [self._pos_agent[0] + 1, self._pos_agent[1]] not in self._pos_walls:
                self._pos_agent[0] = self._pos_agent[0] + 1
        elif action == 1:
            if [self._pos_agent[0], self._pos_agent[1] + 1] not in self._pos_walls:
                self._pos_agent[1] = self._pos_agent[1] + 1
        elif action == 2:
            if [self._pos_agent[0] - 1, self._pos_agent[1]] not in self._pos_walls:
                self._pos_agent[0] = self._pos_agent[0] - 1
        elif action == 3:
            if [self._pos_agent[0], self._pos_agent[1] - 1] not in self._pos_walls:
                self._pos_agent[1] = self._pos_agent[1] - 1

        if self._pos_agent in self._pos_rewards:
            reward = 1
            self._pos_rewards.remove(self._pos_agent)

        self._mode_score += reward
        return reward

    def summarizePerformance(self, test_data_set, learning_algo, *args, **kwargs):
        print("test_data_set.observations.shape")
        print(test_data_set.observations()[0][0:1])

        print("self._mode_score:" + str(self._mode_score) + ".")

    def inputDimensions(self):
        if self._higher_dim_obs == True:
            return [(1, self._size_maze * 6, self._size_maze * 6)]
        else:
            return [(1, self._size_maze, self._size_maze)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return len(self._actions)

    def observe(self):
        self._map = np.zeros((self._size_maze, self._size_maze))
        for coord_wall in self._pos_walls:
            self._map[coord_wall[0], coord_wall[1]] = 1
        for coord_reward in self._pos_rewards:
            self._map[coord_reward[0], coord_reward[1]] = 2
        self._map[self._pos_agent[0], self._pos_agent[1]] = 0.5

        if self._higher_dim_obs == True:
            indices_reward = np.argwhere(self._map == 2)
            indices_agent = np.argwhere(self._map == 0.5)
            self._map = self._map / 1.0
            self._map = np.repeat(np.repeat(self._map, 6, axis=0), 6, axis=1)
            # agent repr
            agent_obs = np.zeros((6, 6))
            agent_obs[0, 2] = 0.8
            agent_obs[1, 0:5] = 0.9
            agent_obs[2, 1:4] = 0.9
            agent_obs[3, 1:4] = 0.9
            agent_obs[4, 1] = 0.9
            agent_obs[4, 3] = 0.9
            agent_obs[5, 0:2] = 0.9
            agent_obs[5, 3:5] = 0.9

            # reward repr
            reward_obs = np.zeros((6, 6))
            reward_obs[:, 1] = 0.7
            reward_obs[0, 1:4] = 0.6
            reward_obs[1, 3] = 0.7
            reward_obs[2, 1:4] = 0.6
            reward_obs[4, 2] = 0.7
            reward_obs[5, 2:4] = 0.7

            for i in indices_reward:
                self._map[
                    i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6
                ] = reward_obs

            for i in indices_agent:
                self._map[
                    i[0] * 6 : (i[0] + 1) * 6 :, i[1] * 6 : (i[1] + 1) * 6
                ] = agent_obs
            self._map = (self._map * 2) - 1  # scaling
            # print ("self._map higher_dim_obs")
            # print (self._map)
            # plt.imshow(self._map, cmap='gray_r')
            # plt.show()
        else:
            self._map = self._map / 2.0
            self._map[self._map == 0.5] = 0.99  # agent
            self._map[self._map == 1.0] = 0.5  # reward

        if self._reverse == True:
            self._map = -self._map  # 1-self._map

        return [self._map]

    def inTerminalState(self):
        if self._pos_rewards == [] or (self._mode >= 0 and self._episode_steps >= 50):
            return True
        else:
            return False


if __name__ == "__main__":
    import hashlib

    rng = np.random.RandomState(123456)
    env = MyEnv(rng, higher_dim_obs=False)

    maps = []
    for i in range(10000):
        env.create_map()

        one_laby = env.observe()[0]

        # Hashing the labyrinths to be able to find duplicates in O(1)
        one_laby = int(hashlib.sha1(str(one_laby).encode("utf-8")).hexdigest(), 16) % (
            10 ** 8
        )

        # TESTING ADDING DUPLICATION
        if i % 1000 == 0:
            env.reset(0)
        if i % 1000 == 500:
            env.reset(1)

        maps.append(copy.deepcopy(one_laby))

    duplicate_laby = 0
    for i in range(10000):
        env.create_map()
        one_laby = env.observe()[0]

        # Hashing the labyrinths to be able to find duplicates in O(1)
        one_laby = int(hashlib.sha1(str(one_laby).encode("utf-8")).hexdigest(), 16) % (
            10 ** 8
        )

        # TESTING ADDING DUPLICATION
        # if i%1000==0:
        #    maps.append(one_laby)

        # TESTING WITH RESETS
        if i % 1000 == 0:
            env.reset(0)
        if i % 1000 == 500:
            env.reset(1)

        duplicate = min(maps.count(one_laby), 1)
        duplicate_laby += duplicate

        if i % 1000 == 0:
            print("Number of duplicate labyrinths:" + str(duplicate_laby) + ".")
