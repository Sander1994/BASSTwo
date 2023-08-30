import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from numpy import ndarray


# GanzSchonClever environment class
class GanzSchonCleverEnv(gym.Env):
    valid_action_mask_value: ndarray
    metadata = {'render.modes': ['human']}

    def __init__(self, rounds=10):
        super(GanzSchonCleverEnv, self).__init__()
        self.initial_rounds = rounds
        self.rounds = rounds
        self.number_of_actions = 16
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.dice = self.roll_dice()
        self.rewards = {'row': [10, 14, 16, 20], 'col': [10, 14, 16, 20]}
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.score_history = []
        self.last_dice = None
        self.extra_pick_unlocked = False

        low_bound = np.array([0]*16 + [1]*4 + [0])
        high_bound = np.array([6]*16 + [6]*4 + [10])
        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(low_bound, high_bound, shape=(21,), dtype=np.int32)
        self.valid_action_mask_value = np.ones(self.number_of_actions)
        self.valid_action_mask_value = self.valid_action_mask()

    # executing actions and returning observations
    def step(self, action):
        reward = 0
        terminated = False
        truncated = False
        info = {}

        if action < 16:
            row = action // 4
            col = action % 4
            if self.yellow_field[row][col] in self.dice and self.yellow_field[row][col] != 0:
                self.yellow_field[row][col] = 0
                reward = self.check_rewards()
            else:
                self.rounds -= 1
                reward -= 1
                if self.rounds == 0:
                    terminated = True
                return self._get_obs(), reward, terminated, truncated, info

        # process for the extra pick reward
        # elif action < 32:
        #     action -= 16
        #     row = action // 4
        #     col = action % 4
        #     if self.extra_pick:
        #         if self.yellow_field[row][col] in self.last_dice and self.yellow_field[row][col] != 0:
        #             self.yellow_field[row][col] = 0
        #             reward = self.check_rewards()
        #             if reward == 0:
        #                 reward += 1
        #             self.score += reward
        #             return self._get_obs(), reward, terminated, truncated, info
        #         else:
        #             reward -= 1
        #             return self._get_obs(), reward, terminated, truncated, info
        #     else:
        #         reward -= 1
        #         return self._get_obs(), reward, terminated, truncated, info

        else:
            reward -= 1000
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info

        self.last_dice = self.dice
        self.dice = self.roll_dice()
        self.rounds -= 1
        self.score += reward
        if self.rounds == 0:
            terminated = True
        self.valid_action_mask_value = self.valid_action_mask()
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    # resetting the environment
    def reset(self, seed=None, **kwargs):
        self.score_history.append(self.score)
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.rounds = self.initial_rounds
        self.dice = self.roll_dice()
        self.extra_pick_unlocked = False
        self.valid_action_mask_value = self.valid_action_mask()
        info = {}

        return self._get_obs(), info

    # rendering the process
    def render(self, mode='human'):
        if mode == 'human':
            print(f'Yellow Field: {self.yellow_field}')
            print(f'Dice: {self.dice}')
            print(f'Score: {self.score}')
        elif mode == 'rgb_array':
            raise NotImplementedError('rgb_array mode is not supported')
        else:
            raise ValueError(f'Render mode {mode} is not supported')

    # rolling the dice
    @staticmethod
    def roll_dice():
        return random.randint(1, 6), random.randint(1, 6), random.randint(1, 6), random.randint(1, 6)

    # checking the rewards for the current step
    def check_rewards(self):
        reward = 0
        for i in range(4):
            if all(self.yellow_field[i][j] == 0 for j in range(4)) and not self.reward_flags['row'][i]:
                reward += self.rewards['row'][i]
                self.reward_flags['row'][i] = True
            if all(self.yellow_field[j][i] == 0 for j in range(4)) and not self.reward_flags['col'][i]:
                reward += self.rewards['col'][i]
                self.reward_flags['col'][i] = True
        if all(self.yellow_field[i][i] == 0 for i in range(4)) and not self.extra_pick and not self.extra_pick_unlocked:
            self.extra_pick = True
            self.extra_pick_unlocked = True
        return reward

    # returning current observations
    def _get_obs(self):
        yellow_field_array = np.array(self.yellow_field, dtype=np.int32).flatten()
        dice_array = np.array(list(self.dice), dtype=np.int32)
        obs = np.concatenate((yellow_field_array, dice_array, [self.rounds]), axis=None)
        return obs

    # returning current action mask
    def valid_action_mask(self) -> np.ndarray:
        self.valid_action_mask_value[:] = 1
        for i in range(self.number_of_actions):
            row = i % 4
            col = int(i / 4)
            if self.yellow_field[col][row] == 0:
                self.valid_action_mask_value[i] = 0
        if 1 not in self.dice:
            self.valid_action_mask_value[5] = 0
            self.valid_action_mask_value[8] = 0
        if 2 not in self.dice:
            self.valid_action_mask_value[4] = 0
            self.valid_action_mask_value[10] = 0
        if 3 not in self.dice:
            self.valid_action_mask_value[0] = 0
            self.valid_action_mask_value[13] = 0
        if 4 not in self.dice:
            self.valid_action_mask_value[11] = 0
            self.valid_action_mask_value[14] = 0
        if 5 not in self.dice:
            self.valid_action_mask_value[2] = 0
            self.valid_action_mask_value[7] = 0
        if 6 not in self.dice:
            self.valid_action_mask_value[1] = 0
            self.valid_action_mask_value[15] = 0
        return self.valid_action_mask_value
