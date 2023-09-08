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
        self.number_of_actions = 28
        self.dice = self.roll_dice()
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.yellow_rewards = {'row': [10, 14, 16, 20], 'col': [10, 14, 16, 20], 'dia': "+1"}
        self.yellow_reward_flags = {'row': [False] * 4, 'col': [False] * 4,
                                    'dia': False}
        self.blue_field = [[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.blue_rewards = {"row": [14, 16, 20], "col": [10, 14, 14, 16]}
        self.blue_reward_flags = {"row": [False] * 3, "col": [False] * 4}
        self.blue_count_rewards = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.blue_count_reward_flags = [False] * 12
        self.extra_pick = False
        self.score = 0
        self.score_history = []
        self.last_dice = None
        self.extra_pick_unlocked = False

        low_bound = np.array([0]*16 + [0]*12 + [1]*6 + [0])
        high_bound = np.array([6]*16 + [6]*12 + [6]*6 + [10])
        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(low_bound, high_bound, shape=(35,), dtype=np.int8)
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
            if self.yellow_field[row][col] in self.dice.values() and self.yellow_field[row][col] != 0:
                self.yellow_field[row][col] = 0
                reward = self.check_rewards()
            else:
                self.rounds -= 1
                reward -= 1
                if self.rounds == 0:
                    terminated = True
                return self._get_obs(), reward, terminated, truncated, info

        elif action < 28:
            action -= 15
            if action == self.dice["white"] + self.dice["blue"] and \
                    any(action in sublist for sublist in self.blue_field):
                for row_index, row in enumerate(self.blue_field):
                    for col_index, element in enumerate(row):
                        if element == action:
                            self.blue_field[row_index][col_index] = 0
                            reward = self.check_rewards()
            else:
                self.rounds -= 1
                reward -= 1
                if self.rounds == 0:
                    terminated = True
                return self._get_obs(), reward, terminated, truncated, info

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
        self.yellow_reward_flags = {'row': [False] * 4, 'col': [False] * 4}
        self.blue_field = [[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.blue_reward_flags = {"row": [False] * 3, "col": [False] * 4}
        self.blue_count_reward_flags = [False] * 12
        self.extra_pick = False
        self.score = 0
        self.rounds = self.initial_rounds
        self.dice = self.roll_dice()
        self.extra_pick_unlocked = False
        self.valid_action_mask_value = self.valid_action_mask()
        info = {}

        return self._get_obs(), info

    # rendering the process
    def render(self, mode="human"):
        if mode == 'human':
            print(f'Yellow Field: {self.yellow_field}')
            print(f'Blue Field: {self.blue_field}')
            print(f'Dice: {self.dice}')
            print(f'Score: {self.score}')
        elif mode == 'rgb_array':
            raise NotImplementedError('rgb_array mode is not supported')
        else:
            raise ValueError(f'Render mode {self.render_mode} is not supported')

    # rolling the dice
    @staticmethod
    def roll_dice():
        colors = ["white", "yellow", "blue", "green", "orange", "purple"]
        return {color: random.randint(1, 6) for color in colors}

    # checking the rewards for the current step
    def check_rewards(self):
        reward = 0
        blue_field_count = 0

        # yellow field rewards
        for i in range(4):
            if all(self.yellow_field[i][j] == 0 for j in range(4)) and not self.yellow_reward_flags['row'][i]:
                reward += self.yellow_rewards['row'][i]
                self.yellow_reward_flags['row'][i] = True
            if all(self.yellow_field[j][i] == 0 for j in range(4)) and not self.yellow_reward_flags['col'][i]:
                reward += self.yellow_rewards['col'][i]
                self.yellow_reward_flags['col'][i] = True
        if all(self.yellow_field[i][i] == 0 for i in range(4)) and not self.extra_pick and not self.extra_pick_unlocked:
            self.extra_pick = True
            self.extra_pick_unlocked = True

        # blue field rewards
        for row in self.blue_field:
            for element in row:
                if element == 0:
                    blue_field_count += 1
        if not self.blue_count_reward_flags[blue_field_count - 1]:
            reward += self.blue_count_rewards[blue_field_count - 1]
            self.blue_count_reward_flags[blue_field_count-1] = True
        for i in range(len(self.blue_field)):
            if all(element == 0 for element in self.blue_field[i]) and not self.blue_reward_flags["row"][i]:
                reward += self.blue_reward_flags["row"][i]
                self.blue_reward_flags["row"][i] = True
        for i, row in self.blue_field:
            if all(row[i] == 0 for row in self.blue_field) and not self.blue_reward_flags["col"][i]:
                reward += self.blue_rewards["col"][i]
                self.blue_reward_flags["col"][i] = True

        return reward

    # returning current observations
    def _get_obs(self):
        yellow_field_array = np.array(self.yellow_field, dtype=np.int8).flatten()
        blue_field_array = np.array(self.blue_field, dtype=np.int8).flatten()
        dice_array = np.array(list(self.dice.values()), dtype=np.int8)
        obs = np.concatenate((yellow_field_array, blue_field_array, dice_array, [self.rounds]), axis=None)
        return obs

    # returning current action mask
    def valid_action_mask(self) -> np.ndarray:
        self.valid_action_mask_value[:] = 1

        # mask for yellow_field_actions
        for i in range(16):
            row = i % 4
            col = int(i / 4)
            if self.yellow_field[col][row] == 0:
                self.valid_action_mask_value[i] = 0
        if 1 not in self.dice.values():
            self.valid_action_mask_value[5] = 0
            self.valid_action_mask_value[8] = 0
        if 2 not in self.dice.values():
            self.valid_action_mask_value[4] = 0
            self.valid_action_mask_value[10] = 0
        if 3 not in self.dice.values():
            self.valid_action_mask_value[0] = 0
            self.valid_action_mask_value[13] = 0
        if 4 not in self.dice.values():
            self.valid_action_mask_value[11] = 0
            self.valid_action_mask_value[14] = 0
        if 5 not in self.dice.values():
            self.valid_action_mask_value[2] = 0
            self.valid_action_mask_value[7] = 0
        if 6 not in self.dice.values():
            self.valid_action_mask_value[1] = 0
            self.valid_action_mask_value[15] = 0

        # mask for blue_field_actions
        m = 16
        for row in self.blue_field:
            for element in row:
                if element == 0:
                    self.valid_action_mask_value[m] = 0
                m += 1
        if self.dice["blue"] + self.dice["white"] != 2:
            self.valid_action_mask_value[17] = 0
        if self.dice["blue"] + self.dice["white"] != 3:
            self.valid_action_mask_value[18] = 0
        if self.dice["blue"] + self.dice["white"] != 4:
            self.valid_action_mask_value[19] = 0
        if self.dice["blue"] + self.dice["white"] != 5:
            self.valid_action_mask_value[20] = 0
        if self.dice["blue"] + self.dice["white"] != 6:
            self.valid_action_mask_value[21] = 0
        if self.dice["blue"] + self.dice["white"] != 7:
            self.valid_action_mask_value[22] = 0
        if self.dice["blue"] + self.dice["white"] != 8:
            self.valid_action_mask_value[23] = 0
        if self.dice["blue"] + self.dice["white"] != 9:
            self.valid_action_mask_value[24] = 0
        if self.dice["blue"] + self.dice["white"] != 10:
            self.valid_action_mask_value[25] = 0
        if self.dice["blue"] + self.dice["white"] != 11:
            self.valid_action_mask_value[26] = 0
        if self.dice["blue"] + self.dice["white"] != 12:
            self.valid_action_mask_value[27] = 0

        return self.valid_action_mask_value
