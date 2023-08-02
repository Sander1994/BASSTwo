import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class GanzSchonCleverEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, rounds=10):
        super(GanzSchonCleverEnv, self).__init__()
        self.initial_rounds = rounds
        self.rounds = rounds
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.dice = self.roll_dice()
        self.rewards = {'row': [10, 10, 10, 25], 'col': [10, 14, 16, 20]}
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0

        self.action_space = spaces.MultiDiscrete([4, 4, 2])  # 4 rows, 4 columns, and binary flag for extra pick
        self.observation_space = spaces.Box(low=0, high=6, shape=(4 * 4 + 2,), dtype=np.int32)

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

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.rounds = self.initial_rounds
        self.dice = self.roll_dice()

        return self._get_obs(), {}

    def render(self, mode='human'):
        if mode == 'human':
            print(f'Yellow Field: {self.yellow_field}')
            print(f'Dice: {self.dice}')
            print(f'Score: {self.score}')
        elif mode == 'rgb_array':
            raise NotImplementedError('rgb_array mode is not supported')
        else:
            raise ValueError(f'Render mode {mode} is not supported')

    @staticmethod
    def roll_dice():
        return random.randint(1, 6), random.randint(1, 6)

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

    def _get_obs(self):
        yellow_field_array = np.array(self.yellow_field, dtype=np.int32).flatten()
        dice_array = np.array(list(self.dice), dtype=np.int32)

        print(f'Yellow Field Array Shape: {yellow_field_array.shape}')
        print(f'Dice Array Shape: {dice_array.shape}')

        obs = np.concatenate((yellow_field_array, dice_array), axis=None)

        print(f'Observation Shape: {obs.shape}')
        return obs

