import gymnasium as gym
from gymnasium import spaces
import numpy as np


class GanzSchonCleverEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rounds=10):
        super(GanzSchonCleverEnv, self).__init__()

        self.initial_rounds = rounds
        self.rounds = rounds
        self.yellow_field = np.array([[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]])
        self.dice = self.roll_dice()
        self.rewards = {'row': [10, 10, 10, 25], 'col': [10, 14, 16, 20]}
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.state = {'field': self.yellow_field.copy(), 'dice': self.dice}

        self.action_space = spaces.MultiDiscrete([4, 4, 2])  # 4 rows, 4 columns, and binary flag for extra pick
        self.observation_space = spaces.Dict({
            'field': spaces.Box(low=0, high=6, shape=(4, 4), dtype=np.int32),
            'dice': spaces.Box(low=1, high=6, shape=(2,), dtype=np.int32)
        })

    def step(self, action):
        row, col, extra_pick_action = action

        reward = 0
        terminated = False
        truncated = False
        info = {}

        # check if the action is valid
        if self.yellow_field[row][col] in self.dice and self.yellow_field[row][col] != 0:
            self.yellow_field[row][col] = 0
            reward = self.check_rewards()
        else:
            terminated = True  # end episode if invalid field action is taken
            return {'field': self.yellow_field.copy(), 'dice': self.dice}, reward, terminated, truncated, info

        # check if extra pick action is valid
        if extra_pick_action == 1:
            if self.extra_pick:
                # find an unentered field that matches a die value
                for i in range(4):
                    for j in range(4):
                        if self.yellow_field[i][j] in self.dice and self.yellow_field[i][j] != 0:
                            self.yellow_field[i][j] = 0
                            reward += self.check_rewards()
                            self.extra_pick = False
                            break
                    if not self.extra_pick:
                        break
                if self.extra_pick:
                    terminated = True  # end episode if invalid extra pick action is taken
                    return {'field': self.yellow_field.copy(), 'dice': self.dice}, reward, terminated, truncated, info
            else:
                terminated = True  # end episode if invalid extra pick action is taken
                return {'field': self.yellow_field.copy(), 'dice': self.dice}, reward, terminated, truncated, info

        # roll the dice for the next state
        self.dice = self.roll_dice()

        # check if all rounds are played
        self.rounds -= 1
        self.score += reward
        if self.rounds == 0:
            terminated = True

        self.state = {self.yellow_field.copy(), [2, 2]}
        return np.array(self.state, dtype=np.float32), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.yellow_field = np.array([[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]])
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.rounds = self.initial_rounds
        self.dice = self.roll_dice()
        self.state = {self.yellow_field.copy(), [2, 2]}

        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        print(f'Yellow Field: {self.yellow_field}')
        print(f'Dice: {self.dice}')

    @staticmethod
    def roll_dice():
        return np.array([np.random.randint(1, 7), np.random.randint(1, 7)])

    def check_rewards(self):
        reward = 0
        for i in range(4):
            if all(self.yellow_field[i][j] == 0 for j in range(4)) and not self.reward_flags['row'][i]:
                reward += self.rewards['row'][i]
                self.reward_flags['row'][i] = True
            if all(self.yellow_field[j][i] == 0 for j in range(4)) and not self.reward_flags['col'][i]:
                reward += self.rewards['col'][i]
                self.reward_flags['col'][i] = True
        if all(self.yellow_field[i][i] == 0 for i in range(4)) and not self.extra_pick:
            self.extra_pick = True
        return reward
