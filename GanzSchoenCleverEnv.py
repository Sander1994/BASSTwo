import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
from numpy import ndarray


# GanzSchonClever environment class
class GanzSchonCleverEnv(gym.Env):
    valid_action_mask_value: ndarray
    metadata = {'render.modes': ['human']}

    def __init__(self, rounds=6, render_mode="human"):
        super(GanzSchonCleverEnv, self).__init__()
        # initial board values
        self.render_mode = render_mode
        self.initial_rounds = rounds
        self.rounds = rounds
        self.roll_in_round = 1
        self.invalid_dice = {"white": False, "yellow": False, "blue": False, "green": False, "orange": False,
                             "purple": False}
        self.dice = {"white": 0, "yellow": 0, "blue": 0, "green": 0, "orange": 0, "purple": 0}
        self.roll_dice()
        self.can_pick_extra_self = False
        self.can_pick_other = False
        self.can_pick_extra_other = False
        self.turn_is_extra_turn = False
        self.score = 0
        self.score_history = []
        self.initialized = False
        # fields
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.yellow_rewards = {"row": ["blue_cross", "orange_four", "green_cross", "fox"], "col": [10, 14, 16, 20],
                               "dia": "extra_pick"}
        self.yellow_reward_flags = {"row": [False] * 4, "col": [False] * 4, "dia": False}
        self.yellow_field_score = 0
        self.blue_field = [[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.blue_rewards = {"row": ["orange_five", "yellow_cross", "fox"],
                             "col": ["re_roll", "green_cross", "purple_six", "extra_pick"]}
        self.blue_reward_flags = {"row": [False] * 3, "col": [False] * 4}
        self.blue_count_rewards = [0, 1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.blue_count_reward_flags = [False] * 12
        self.blue_field_score = 0
        self.green_field = [0] * 11
        self.green_rewards = [None, None, None, "extra_pick", None, "blue_cross", "fox", None, "purple_six", "re_roll",
                              None]
        self.green_count_rewards = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        self.green_reward_flags = [False] * 11
        self.green_field_score = 0
        self.orange_field = [0] * 11
        self.orange_rewards = [None, None, "re_roll", None, "yellow_cross", "extra_pick", None, "fox", None,
                               "purple_six", None]
        self.orange_reward_flags = [False] * 11
        self.orange_field_score = 0
        self.purple_field = [0] * 11
        self.purple_rewards = [None, None, "re_roll", "blue_cross", "extra_pick", "yellow_cross", "fox", "re_roll",
                               "green_cross", "orange_six", "extra_pick"]
        self.purple_reward_flags = [False] * 11
        self.purple_field_score = 0
        # rewards
        self.extra_pick = 0
        self.re_roll = 1
        self.fox = 0
        self.yellow_cross = 0
        self.blue_cross = 0
        self.green_cross = 0
        self.orange_four = 0
        self.orange_five = 0
        self.orange_six = 0
        self.purple_six = 0
        # picks
        self.picked_yellow = 0
        self.picked_blue = 0
        self.picked_green = 0
        self.picked_orange = 0
        self.picked_purple = 0
        # model values
        self.number_of_actions = 247
        low_bound = np.array([0]*16 + [0]*12 + [0] * 11 + [0] * 11 + [0] * 11 + [1]*6 + [0]*6 + [0] + [1] + [0] * 3 +
                             [0] * 7)
        high_bound = np.array([6]*16 + [6]*12 + [1] * 11 + [6] * 11 + [6] * 11 + [6]*6 + [1]*6 + [6] + [3] + [6] * 3 +
                              [1] * 7)
        self.action_space = spaces.Discrete(self.number_of_actions)
        self.observation_space = spaces.Box(low_bound, high_bound, shape=(85,), dtype=np.int8)
        self.valid_action_mask_value = np.ones(self.number_of_actions)
        self.valid_action_mask_value = self.valid_action_mask()

    # executing actions and returning observations
    def step(self, action):
        # initializing values
        reward = 0
        terminated = False
        truncated = False
        info = {}
        self.turn_is_extra_turn = False
        # yellow field actions
        if action < 16:
            self.picked_yellow += 1
            if self.yellow_cross >= 1:
                self.turn_is_extra_turn = True
                self.yellow_cross -= 1
            row = action // 4
            col = action % 4
            self.yellow_field[row][col] = 0
            if not self.turn_is_extra_turn:
                for dice_color in self.dice:
                    if self.dice[dice_color] < self.dice["yellow"]:
                        self.invalid_dice[dice_color] = True
                self.invalid_dice["yellow"] = True
        # blue field actions
        elif action < 28:
            self.picked_blue += 1
            action -= 15
            if self.blue_cross >= 1:
                self.turn_is_extra_turn = True
                self.blue_cross -= 1
            for row_index, row in enumerate(self.blue_field):
                for col_index, element in enumerate(row):
                    if element == action:
                        self.blue_field[row_index][col_index] = 0
            if not self.turn_is_extra_turn:
                for dice_color in self.dice:
                    if self.dice[dice_color] < self.dice["blue"]:
                        self.invalid_dice[dice_color] = True
                self.invalid_dice["blue"] = True
        # green field actions
        elif action < 39:
            self.picked_green += 1
            action -= 28
            if self.green_cross >= 1:
                self.turn_is_extra_turn = True
                self.green_cross -= 1
            self.green_field[action] = 1
            if not self.turn_is_extra_turn:
                for dice_color in self.dice:
                    if self.dice[dice_color] < self.dice["green"]:
                        self.invalid_dice[dice_color] = True
                self.invalid_dice["green"] = True
        # orange field actions
        elif action < 50:
            self.picked_orange += 1
            action -= 39
            if self.orange_four >= 1:
                self.turn_is_extra_turn = True
                self.orange_field[action] = 4
                self.orange_four -= 1
            elif self.orange_five >= 1:
                self.turn_is_extra_turn = True
                self.orange_field[action] = 5
                self.orange_five -= 1
            elif self.orange_six >= 1:
                self.turn_is_extra_turn = True
                self.orange_field[action] = 6
                self.orange_six -= 1
            else:
                self.orange_field[action] = self.dice["orange"]
            if not self.turn_is_extra_turn:
                for dice_color in self.dice:
                    if self.dice[dice_color] < self.dice["orange"]:
                        self.invalid_dice[dice_color] = True
                self.invalid_dice["orange"] = True
        # purple field actions
        elif action < 61:
            self.picked_purple += 1
            action -= 50
            if self.purple_six >= 1:
                self.turn_is_extra_turn = True
                self.purple_field[action] = 6
                self.purple_six -= 1
            else:
                self.purple_field[action] = self.dice["purple"]
            if not self.turn_is_extra_turn:
                for dice_color in self.dice:
                    if self.dice[dice_color] < self.dice["purple"]:
                        self.invalid_dice[dice_color] = True
                self.invalid_dice["purple"] = True
        # white dice actions
        elif action < 122:
            # yellow
            if action < 77:
                action -= 61
                self.picked_yellow += 1
                row = action // 4
                col = action % 4
                self.yellow_field[row][col] = 0
            # blue
            elif action < 89:
                self.picked_blue += 1
                action -= 76
                for row_index, row in enumerate(self.blue_field):
                    for col_index, element in enumerate(row):
                        if element == action:
                            self.blue_field[row_index][col_index] = 0
            # green
            elif action < 100:
                self.picked_green += 1
                action -= 89
                self.green_field[action] = 1
            # orange
            elif action < 111:
                self.picked_orange += 1
                action -= 100
                self.orange_field[action] = self.dice["white"]
            # purple
            elif action < 122:
                self.picked_purple += 1
                action -= 111
                self.purple_field[action] = self.dice["white"]
            # setting invalid dice
            for dice_color in self.dice:
                if self.dice[dice_color] < self.dice["white"]:
                    self.invalid_dice[dice_color] = True
            self.invalid_dice["white"] = True
        # extra_pick action
        elif action < 244:
            if self.can_pick_extra_self or self.can_pick_extra_other:
                self.extra_pick -= 1
            self.turn_is_extra_turn = True
            # yellow field actions
            if action < 138:
                self.picked_yellow += 1
                action -= 122
                row = action // 4
                col = action % 4
                self.yellow_field[row][col] = 0
                self.invalid_dice["yellow"] = True
            # blue field actions
            elif action < 150:
                self.picked_blue += 1
                action -= 137
                for row_index, row in enumerate(self.blue_field):
                    for col_index, element in enumerate(row):
                        if element == action:
                            self.blue_field[row_index][col_index] = 0
                self.invalid_dice["blue"] = True
            # green field actions
            elif action < 161:
                self.picked_green += 1
                action -= 150
                self.green_field[action] = 1
                self.invalid_dice["green"] = True
            # orange field actions
            elif action < 172:
                self.picked_orange += 1
                action -= 161
                self.orange_field[action] = self.dice["orange"]
                self.invalid_dice["orange"] = True
            # purple field actions
            elif action < 183:
                self.picked_purple += 1
                action -= 172
                self.purple_field[action] = self.dice["purple"]
                self.invalid_dice["purple"] = True
            # white dice actions
            elif action < 244:
                # yellow
                if action < 199:
                    action -= 183
                    self.picked_yellow += 1
                    row = action // 4
                    col = action % 4
                    self.yellow_field[row][col] = 0
                # blue
                elif action < 211:
                    self.picked_blue += 1
                    action -= 198
                    for row_index, row in enumerate(self.blue_field):
                        for col_index, element in enumerate(row):
                            if element == action:
                                self.blue_field[row_index][col_index] = 0
                # green
                elif action < 222:
                    self.picked_green += 1
                    action -= 211
                    self.green_field[action] = 1
                # orange
                elif action < 233:
                    self.picked_orange += 1
                    action -= 222
                    self.orange_field[action] = self.dice["white"]
                # purple
                elif action < 244:
                    self.picked_purple += 1
                    action -= 233
                    self.purple_field[action] = self.dice["white"]
                self.invalid_dice["white"] = True
            if self.extra_pick <= 0 or self.can_pick_other:
                self.increment_rounds()
        # re_roll action
        elif action < 245:
            self.roll_dice()
            self.re_roll -= 1
            self.valid_action_mask_value = self.valid_action_mask()
            return self._get_obs(), reward, terminated, truncated, info
        # do nothing instead of extra_pick
        elif action < 246:
            self.increment_rounds()
            return self._get_obs(), reward, terminated, truncated, info
        # increment rounds if no action is possible
        elif action < 247:
            reward -= 1
        # wrong actions
        else:
            reward -= 1000
            terminated = True
            return self._get_obs(), reward, terminated, truncated, info
        # attribute updates
        reward += self.check_rewards()
        if not self.turn_is_extra_turn:
            self.increment_rounds()
            # attribute updates for a finished game
            if self.rounds == 0:
                terminated = True
                reward += self.fox * min(self.yellow_field_score, self.blue_field_score, self.green_field_score,
                                         self.orange_field_score, self.purple_field_score)
        if reward >= 0:
            self.score += reward
        self.valid_action_mask_value = self.valid_action_mask()
        info = {}

        return self._get_obs(), reward, terminated, truncated, info

    # resetting the environment
    def reset(self, seed=None, **kwargs):
        if self.initialized is True:
            self.score_history.append(self.score)
        self.invalid_dice = {"white": False, "yellow": False, "blue": False, "green": False, "orange": False,
                             "purple": False}
        self.roll_dice()
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.yellow_reward_flags = {"row": [False] * 4, "col": [False] * 4, "dia": False}
        self.yellow_field_score = 0
        self.blue_field = [[0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        self.blue_reward_flags = {"row": [False] * 3, "col": [False] * 4}
        self.blue_count_reward_flags = [False] * 12
        self.blue_field_score = 0
        self.green_field = [0] * 11
        self.green_reward_flags = [False] * 11
        self.green_field_score = 0
        self.orange_field = [0] * 11
        self.orange_reward_flags = [False] * 11
        self.orange_field_score = 0
        self.purple_field = [0] * 11
        self.purple_reward_flags = [False] * 11
        self.purple_field_score = 0
        self.turn_is_extra_turn = False
        self.can_pick_extra_self = False
        self.can_pick_other = False
        self.can_pick_extra_other = False
        self.extra_pick = 0
        self.re_roll = 1
        self.fox = 0
        self.yellow_cross = 0
        self.blue_cross = 0
        self.green_cross = 0
        self.orange_four = 0
        self.orange_five = 0
        self.orange_six = 0
        self.purple_six = 0
        self.score = 0
        self.rounds = self.initial_rounds
        self.roll_in_round = 1
        self.valid_action_mask_value = self.valid_action_mask()
        info = {}
        self.initialized = True

        return self._get_obs(), info

    # rendering the process
    def render(self):
        if self.render_mode == 'human':
            print(f'Yellow Field: {self.yellow_field}')
            print(f'Blue Field: {self.blue_field}')
            print(f'Green Field: {self.green_field}')
            print(f'Orange Field: {self.orange_field}')
            print(f'Purple Field: {self.purple_field}')
            print(f'Dice: {self.dice}')
            print(f'Invalid Dice: {self.invalid_dice}')
            print(f'Action Mask: {self.valid_action_mask_value}')
            print(f'Score: {self.score}')
            print(f'Score History: {self.score_history}')
            if self.score_history:
                print(f'Max Score: {max(self.score_history)}')
                print(f'Average Score: {sum(self.score_history) / len(self.score_history)}')
                print(f'Standard Deviation: {np.std(self.score_history)}')
                self.print_median_score()
            print(f'Picked Colors: Yellow: {self.picked_yellow}, Blue: {self.picked_blue}, Green: {self.picked_green}, '
                  f'Orange: {self.picked_orange}, Purple: {self.picked_purple}')
        elif self.render_mode == 'rgb_array':
            raise NotImplementedError('rgb_array mode is not supported')
        else:
            raise ValueError(f'Render mode {self.render_mode} is not supported')

    # rolling the dice
    def roll_dice(self):
        colors = ["white", "yellow", "blue", "green", "orange", "purple"]
        for color in colors:
            if not self.invalid_dice[color]:
                self.dice[color] = random.randint(1, 6)

    # checking the rewards for the current step
    def check_rewards(self):
        reward = 0
        blue_field_count = 0
        # yellow field rewards
        for i in range(4):
            if all(self.yellow_field[i][j] == 0 for j in range(4)) and not self.yellow_reward_flags["row"][i]:
                self.add_reward(self.yellow_rewards["row"][i])
                self.yellow_reward_flags["row"][i] = True
            if all(self.yellow_field[j][i] == 0 for j in range(4)) and not self.yellow_reward_flags["col"][i]:
                reward += self.yellow_rewards["col"][i]
                self.yellow_field_score += self.yellow_rewards["col"][i]
                self.yellow_reward_flags['col'][i] = True
        if all(self.yellow_field[i][i] == 0 for i in range(4)) and not self.yellow_reward_flags["dia"]:
            self.add_reward(self.yellow_rewards["dia"])
            self.yellow_reward_flags["dia"] = True
        # blue field rewards
        for row in self.blue_field:
            for element in row:
                if element == 0:
                    blue_field_count += 1
        if not self.blue_count_reward_flags[blue_field_count - 1]:
            reward += self.blue_count_rewards[blue_field_count - 1]
            self.blue_field_score += self.blue_count_rewards[blue_field_count - 1]
            self.blue_count_reward_flags[blue_field_count-1] = True
        for i in range(len(self.blue_field)):
            if all(element == 0 for element in self.blue_field[i]) and not self.blue_reward_flags["row"][i]:
                self.add_reward(self.blue_rewards["row"][i])
                self.blue_reward_flags["row"][i] = True
        for i in range(4):
            if all(row[i] == 0 for row in self.blue_field) and not self.blue_reward_flags["col"][i]:
                self.add_reward(self.blue_rewards["col"][i])
                self.blue_reward_flags["col"][i] = True
        # green field rewards
        for i in range(11):
            if self.green_field[i] == 1 and self.green_reward_flags[i] is False:
                reward += self.green_count_rewards[i]
                self.green_field_score += self.green_count_rewards[i]
                self.add_reward(self.green_rewards[i])
                self.green_reward_flags[i] = True
        # orange field rewards
        for i in range(11):
            if self.orange_field[i] > 0 and self.orange_reward_flags[i] is False:
                reward += self.orange_field[i]
                self.orange_field_score += self.orange_field[i]
                self.add_reward(self.orange_rewards[i])
                self.orange_reward_flags[i] = True
                if i in [3, 6, 8]:
                    reward += self.orange_field[i]
                    self.orange_field_score += self.orange_field[i]
                if i == 10:
                    reward += self.orange_field[i] * 2
                    self.orange_field_score += self.orange_field[i] * 2
        # purple field rewards
        for i in range(11):
            if self.purple_field[i] > 0 and self.purple_reward_flags[i] is False:
                reward += self.purple_field[i]
                self.purple_field_score += self.purple_field[i]
                self.add_reward(self.purple_rewards[i])
                self.purple_reward_flags[i] = True

        return reward

    # returning current observations
    def _get_obs(self):
        # transforming fields to np arrays
        yellow_field_array = np.array(self.yellow_field, dtype=np.int8).flatten()
        blue_field_array = np.array(self.blue_field, dtype=np.int8).flatten()
        green_field_array = np.array(self.green_field, dtype=np.int8).flatten()
        orange_field_array = np.array(self.orange_field, dtype=np.int8).flatten()
        purple_field_array = np.array(self.purple_field, dtype=np.int8).flatten()
        dice_array = np.array(list(self.dice.values()), dtype=np.int8)
        invalid_dice_array = np.array(list(self.invalid_dice.values()), dtype=np.int8)
        # concatenating all values to an observation space
        obs = np.concatenate((yellow_field_array, blue_field_array, green_field_array, orange_field_array,
                              purple_field_array, dice_array, invalid_dice_array, [self.rounds], [self.roll_in_round],
                              [self.extra_pick, self.re_roll, self.fox],
                              [self.orange_four, self.orange_five, self.orange_six, self.purple_six, self.yellow_cross,
                               self.blue_cross, self.green_cross]), axis=None)
        return obs

    # returning current action mask
    def valid_action_mask(self) -> np.ndarray:
        self.valid_action_mask_value[:] = 1
        # mask for yellow field actions
        for i in range(16):
            row = i % 4
            col = int(i / 4)
            if self.yellow_field[col][row] == 0:
                self.valid_action_mask_value[i] = 0
                self.valid_action_mask_value[i + 61] = 0
                self.valid_action_mask_value[122 + i] = 0
                self.valid_action_mask_value[122 + i + 61] = 0
        # yellow dice
        if self.dice["yellow"] != 1:
            self.valid_action_mask_value[5] = 0
            self.valid_action_mask_value[8] = 0
            self.valid_action_mask_value[122 + 5] = 0
            self.valid_action_mask_value[122 + 8] = 0
        if self.dice["yellow"] != 2:
            self.valid_action_mask_value[4] = 0
            self.valid_action_mask_value[10] = 0
            self.valid_action_mask_value[122 + 4] = 0
            self.valid_action_mask_value[122 + 10] = 0
        if self.dice["yellow"] != 3:
            self.valid_action_mask_value[0] = 0
            self.valid_action_mask_value[13] = 0
            self.valid_action_mask_value[122 + 0] = 0
            self.valid_action_mask_value[122 + 13] = 0
        if self.dice["yellow"] != 4:
            self.valid_action_mask_value[11] = 0
            self.valid_action_mask_value[14] = 0
            self.valid_action_mask_value[122 + 11] = 0
            self.valid_action_mask_value[122 + 14] = 0
        if self.dice["yellow"] != 5:
            self.valid_action_mask_value[2] = 0
            self.valid_action_mask_value[7] = 0
            self.valid_action_mask_value[122 + 2] = 0
            self.valid_action_mask_value[122 + 7] = 0
        if self.dice["yellow"] != 6:
            self.valid_action_mask_value[1] = 0
            self.valid_action_mask_value[15] = 0
            self.valid_action_mask_value[122 + 1] = 0
            self.valid_action_mask_value[122 + 15] = 0
        # white dice
        if self.dice["white"] != 1:
            self.valid_action_mask_value[5 + 61] = 0
            self.valid_action_mask_value[8 + 61] = 0
            self.valid_action_mask_value[122 + 5 + 61] = 0
            self.valid_action_mask_value[122 + 8 + 61] = 0
        if self.dice["white"] != 2:
            self.valid_action_mask_value[4 + 61] = 0
            self.valid_action_mask_value[10 + 61] = 0
            self.valid_action_mask_value[122 + 4 + 61] = 0
            self.valid_action_mask_value[122 + 10 + 61] = 0
        if self.dice["white"] != 3:
            self.valid_action_mask_value[0 + 61] = 0
            self.valid_action_mask_value[13 + 61] = 0
            self.valid_action_mask_value[122 + 0 + 61] = 0
            self.valid_action_mask_value[122 + 13 + 61] = 0
        if self.dice["white"] != 4:
            self.valid_action_mask_value[11 + 61] = 0
            self.valid_action_mask_value[14 + 61] = 0
            self.valid_action_mask_value[122 + 11 + 61] = 0
            self.valid_action_mask_value[122 + 14 + 61] = 0
        if self.dice["white"] != 5:
            self.valid_action_mask_value[2 + 61] = 0
            self.valid_action_mask_value[7 + 61] = 0
            self.valid_action_mask_value[122 + 2 + 61] = 0
            self.valid_action_mask_value[122 + 7 + 61] = 0
        if self.dice["white"] != 6:
            self.valid_action_mask_value[1 + 61] = 0
            self.valid_action_mask_value[15 + 61] = 0
            self.valid_action_mask_value[122 + 1 + 61] = 0
            self.valid_action_mask_value[122 + 15 + 61] = 0
        # mask for blue field actions
        m = 16
        for row in self.blue_field:
            for element in row:
                if element == 0:
                    self.valid_action_mask_value[m] = 0
                    self.valid_action_mask_value[m + 61] = 0
                    self.valid_action_mask_value[122 + m] = 0
                    self.valid_action_mask_value[122 + m + 61] = 0
                m += 1
        if self.dice["blue"] + self.dice["white"] != 2:
            self.valid_action_mask_value[17] = 0
            self.valid_action_mask_value[17 + 61] = 0
            self.valid_action_mask_value[122 + 17] = 0
            self.valid_action_mask_value[122 + 17 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 3:
            self.valid_action_mask_value[18] = 0
            self.valid_action_mask_value[18 + 61] = 0
            self.valid_action_mask_value[122 + 17] = 0
            self.valid_action_mask_value[122 + 17 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 4:
            self.valid_action_mask_value[19] = 0
            self.valid_action_mask_value[19 + 61] = 0
            self.valid_action_mask_value[122 + 19] = 0
            self.valid_action_mask_value[122 + 19 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 5:
            self.valid_action_mask_value[20] = 0
            self.valid_action_mask_value[20 + 61] = 0
            self.valid_action_mask_value[122 + 20] = 0
            self.valid_action_mask_value[122 + 20 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 6:
            self.valid_action_mask_value[21] = 0
            self.valid_action_mask_value[21 + 61] = 0
            self.valid_action_mask_value[122 + 21] = 0
            self.valid_action_mask_value[122 + 21 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 7:
            self.valid_action_mask_value[22] = 0
            self.valid_action_mask_value[22 + 61] = 0
            self.valid_action_mask_value[122 + 22] = 0
            self.valid_action_mask_value[122 + 22 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 8:
            self.valid_action_mask_value[23] = 0
            self.valid_action_mask_value[23 + 61] = 0
            self.valid_action_mask_value[122 + 23] = 0
            self.valid_action_mask_value[122 + 23 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 9:
            self.valid_action_mask_value[24] = 0
            self.valid_action_mask_value[24 + 61] = 0
            self.valid_action_mask_value[122 + 24] = 0
            self.valid_action_mask_value[122 + 24 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 10:
            self.valid_action_mask_value[25] = 0
            self.valid_action_mask_value[25 + 61] = 0
            self.valid_action_mask_value[122 + 25] = 0
            self.valid_action_mask_value[122 + 25 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 11:
            self.valid_action_mask_value[26] = 0
            self.valid_action_mask_value[26 + 61] = 0
            self.valid_action_mask_value[122 + 26] = 0
            self.valid_action_mask_value[122 + 26 + 61] = 0
        if self.dice["blue"] + self.dice["white"] != 12:
            self.valid_action_mask_value[27] = 0
            self.valid_action_mask_value[27 + 61] = 0
            self.valid_action_mask_value[122 + 27] = 0
            self.valid_action_mask_value[122 + 27 + 61] = 0
        # mask for green field actions
        m = 0
        self.valid_action_mask_value[28:28 + 11] = 0
        self.valid_action_mask_value[89:89 + 11] = 0
        self.valid_action_mask_value[150:150 + 11] = 0
        self.valid_action_mask_value[211:211 + 11] = 0
        for i in range(len(self.green_field)):
            if self.green_field[i] == 0:
                break
            m += 1
        if m < 11:
            self.valid_action_mask_value[28 + m] = 1
            self.valid_action_mask_value[28 + 61 + m] = 1
            self.valid_action_mask_value[122 + 28 + m] = 1
            self.valid_action_mask_value[122 + 28 + 61 + m] = 1
        # green dice
        if self.dice["green"] == 5:
            self.valid_action_mask_value[28 + 10] = 0
            self.valid_action_mask_value[122 + 28 + 10] = 0
        if self.dice["green"] == 4:
            self.valid_action_mask_value[28 + 9] = 0
            self.valid_action_mask_value[28 + 4] = 0
            self.valid_action_mask_value[122 + 28 + 9] = 0
            self.valid_action_mask_value[122 + 28 + 4] = 0
        if self.dice["green"] == 3:
            self.valid_action_mask_value[28 + 8] = 0
            self.valid_action_mask_value[28 + 3] = 0
            self.valid_action_mask_value[122 + 28 + 8] = 0
            self.valid_action_mask_value[122 + 28 + 3] = 0
        if self.dice["green"] == 2:
            self.valid_action_mask_value[28 + 7] = 0
            self.valid_action_mask_value[28 + 2] = 0
            self.valid_action_mask_value[122 + 28 + 7] = 0
            self.valid_action_mask_value[122 + 28 + 2] = 0
        if self.dice["green"] == 2:
            self.valid_action_mask_value[28 + 6] = 0
            self.valid_action_mask_value[28 + 1] = 0
            self.valid_action_mask_value[122 + 28 + 6] = 0
            self.valid_action_mask_value[122 + 28 + 1] = 0
        # white dice
        if self.dice["white"] == 5:
            self.valid_action_mask_value[28 + 10 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 10 + 61] = 0
        if self.dice["white"] == 4:
            self.valid_action_mask_value[28 + 9 + 61] = 0
            self.valid_action_mask_value[28 + 4 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 9 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 4 + 61] = 0
        if self.dice["white"] == 3:
            self.valid_action_mask_value[28 + 8 + 61] = 0
            self.valid_action_mask_value[28 + 3 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 8 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 3 + 61] = 0
        if self.dice["white"] == 2:
            self.valid_action_mask_value[28 + 7 + 61] = 0
            self.valid_action_mask_value[28 + 2 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 7 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 2 + 61] = 0
        if self.dice["white"] == 2:
            self.valid_action_mask_value[28 + 6 + 61] = 0
            self.valid_action_mask_value[28 + 1 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 6 + 61] = 0
            self.valid_action_mask_value[122 + 28 + 1 + 61] = 0
        # mask for orange field actions
        m = 0
        self.valid_action_mask_value[39:39 + 11] = 0
        self.valid_action_mask_value[100:100 + 11] = 0
        self.valid_action_mask_value[161:161 + 11] = 0
        self.valid_action_mask_value[222:222 + 11] = 0
        for i in range(len(self.orange_field)):
            if self.orange_field[i] == 0:
                break
            m += 1
        if m < 11:
            self.valid_action_mask_value[39 + m] = 1
            self.valid_action_mask_value[39 + 61 + m] = 1
            self.valid_action_mask_value[122 + 39 + m] = 1
            self.valid_action_mask_value[122 + 39 + 61 + m] = 1
        # mask for purple field actions
        m = 0
        self.valid_action_mask_value[50:50 + 11] = 0
        self.valid_action_mask_value[111:111 + 11] = 0
        self.valid_action_mask_value[172:172 + 11] = 0
        self.valid_action_mask_value[233:233 + 11] = 0
        for i in range(len(self.purple_field)):
            if self.purple_field[i] == 0:
                break
            m += 1
        if m < 11 and self.purple_field[m - 1] < self.dice["purple"] or \
                m < 11 and self.purple_field[m - 1] == 6 or \
                m < 11 and self.purple_field[m - 1] < self.dice["white"] or m == 0:
            self.valid_action_mask_value[50 + m] = 1
            self.valid_action_mask_value[50 + 61 + m] = 1
            self.valid_action_mask_value[122 + 50 + m] = 1
            self.valid_action_mask_value[122 + 50 + 61 + m] = 1
        # mask for invalid dice
        if self.invalid_dice["yellow"] is True:
            self.valid_action_mask_value[0:0 + 16] = 0
            self.valid_action_mask_value[122:122 + 16] = 0
        if self.invalid_dice["blue"] is True:
            self.valid_action_mask_value[16:16 + 12] = 0
            self.valid_action_mask_value[138:138 + 12] = 0
        if self.invalid_dice["green"] is True:
            self.valid_action_mask_value[28:28 + 11] = 0
            self.valid_action_mask_value[150:150 + 11] = 0
        if self.invalid_dice["orange"] is True:
            self.valid_action_mask_value[39:39 + 11] = 0
            self.valid_action_mask_value[161:161 + 11] = 0
        if self.invalid_dice["purple"] is True:
            self.valid_action_mask_value[50:50 + 11] = 0
            self.valid_action_mask_value[172:172 + 11] = 0
        if self.invalid_dice["white"] is True:
            self.valid_action_mask_value[61:61 + 61] = 0
            self.valid_action_mask_value[183:183 + 61] = 0

        # mask for extra_pick action
        if not self.can_pick_other and not self.can_pick_extra_self and not self.can_pick_extra_other or \
                (self.can_pick_extra_self and self.extra_pick <= 0) or \
                (self.extra_pick <= 0 and self.can_pick_extra_other):
            self.valid_action_mask_value[122:122 + 122] = 0
            self.valid_action_mask_value[245] = 0
        if (self.extra_pick >= 1 and self.can_pick_extra_self) or (self.extra_pick >= 1 and self.can_pick_extra_other) \
                or self.can_pick_other:
            self.valid_action_mask_value[0:0 + 122] = 0
            if not self.can_pick_other:
                self.valid_action_mask_value[245] = 1
        # mask for re_roll action
        if self.re_roll <= 0 or self.can_pick_extra_self or self.can_pick_extra_other or self.can_pick_other:
            self.valid_action_mask_value[244] = 0

        # mask for rewards
        if self.yellow_cross > 0 or self.blue_cross > 0 or self.green_cross > 0 or self.orange_four > 0 or \
                self.orange_five > 0 or self.orange_six > 0 or self.purple_six > 0:
            self.valid_action_mask_value[:] = 0
        # mask for yellow cross action
        if self.yellow_cross > 0:
            for i in range(16):
                row = i // 4
                col = i % 4
                if self.yellow_field[row][col] > 0:
                    self.valid_action_mask_value[i] = 1
        # mask for blue cross action
        elif self.blue_cross > 0:
            for i in range(12):
                row = i // 4
                col = i % 4
                if self.blue_field[row][col] > 0:
                    self.valid_action_mask_value[16 + i] = 1
        # mask for green cross action
        elif self.green_cross > 0:
            m = 0
            for i in range(len(self.green_field)):
                if self.green_field[i] == 0:
                    break
                m += 1
            if m < 11:
                self.valid_action_mask_value[28 + m] = 1
        # mask for orange four/five/six action
        elif self.orange_four > 0 or self.orange_five > 0 or self.orange_six > 0:
            m = 0
            for i in range(len(self.orange_field)):
                if self.orange_field[i] == 0:
                    break
                m += 1
            if m < 11:
                self.valid_action_mask_value[39 + m] = 1
        # mask for purple six action
        elif self.purple_six > 0:
            m = 0
            for i in range(len(self.purple_field)):
                if self.purple_field[i] == 0:
                    break
                m += 1
            if m < 11:
                self.valid_action_mask_value[50 + m] = 1

        # mask for passing
        if np.all(self.valid_action_mask_value[0:0 + 246] == 0):
            self.valid_action_mask_value[246] = 1
        else:
            self.valid_action_mask_value[246] = 0

        return self.valid_action_mask_value

    # adds the non-numeric rewards
    def add_reward(self, reward_type):
        reward_map = {
            "extra_pick": lambda: setattr(self, "extra_pick", self.extra_pick + 1),
            "re_roll": lambda: setattr(self, "re_roll", self.re_roll + 1),
            "fox": lambda: setattr(self, "fox", self.fox + 1),
            "orange_four": lambda: setattr(self, "orange_four", self.orange_four + 1),
            "orange_five": lambda: setattr(self, "orange_five", self.orange_five + 1),
            "orange_six": lambda: setattr(self, "orange_six", self.orange_six + 1),
            "purple_six": lambda: setattr(self, "purple_six", self.purple_six + 1),
            "yellow_cross": lambda: setattr(self, "yellow_cross", self.yellow_cross + 1),
            "blue_cross": lambda: setattr(self, "blue_cross", self.blue_cross + 1),
            "green_cross": lambda: setattr(self, "green_cross", self.green_cross + 1)
        }
        if reward_type in reward_map:
            reward_map[reward_type]()

    def increment_rounds(self):
        if self.can_pick_extra_self:
            self.can_pick_other = True
            self.can_pick_extra_self = False
            self.invalid_dice = {color: False for color in self.invalid_dice}
            self.roll_dice()
            colors_to_change = random.sample(self.invalid_dice.keys(), 3)
            for color in colors_to_change:
                self.invalid_dice[color] = True
        elif self.can_pick_other:
            if self.extra_pick >= 1:
                self.can_pick_extra_other = True
            else:
                if self.rounds == 5:
                    self.extra_pick += 1
                if self.rounds == 4:
                    self.re_roll += 1
                if self.rounds == 3:
                    random_reward = random.randint(1, 5)
                    if random_reward == 1:
                        self.yellow_cross += 1
                    if random_reward == 2:
                        self.blue_cross += 1
                    if random_reward == 3:
                        self.green_cross += 1
                    if random_reward == 4:
                        self.orange_six += 1
                    if random_reward == 5:
                        self.purple_six += 1
                self.roll_dice()
            self.can_pick_other = False
            self.invalid_dice = {color: False for color in self.invalid_dice}
        elif self.can_pick_extra_other:
            self.can_pick_extra_other = False
            self.roll_dice()
            self.invalid_dice = {color: False for color in self.invalid_dice}
            if self.rounds == 5:
                self.extra_pick += 1
            if self.rounds == 4:
                self.re_roll += 1
            if self.rounds == 3:
                random_reward = random.randint(1, 5)
                if random_reward == 1:
                    self.yellow_cross += 1
                if random_reward == 2:
                    self.blue_cross += 1
                if random_reward == 3:
                    self.green_cross += 1
                if random_reward == 4:
                    self.orange_six += 1
                if random_reward == 5:
                    self.purple_six += 1
        elif self.roll_in_round >= 3:
            self.rounds -= 1
            self.roll_in_round = 1
            self.invalid_dice = {color: False for color in self.invalid_dice}
            if self.extra_pick >= 1:
                self.can_pick_extra_self = True
            else:
                self.can_pick_other = True
                self.roll_dice()
                colors_to_change = random.sample(self.invalid_dice.keys(), 3)
                for color in colors_to_change:
                    self.invalid_dice[color] = True
        elif self.roll_in_round < 3:
            self.roll_in_round += 1
            self.roll_dice()

    def print_median_score(self):
        if self.score_history:
            sorted_scores = sorted(self.score_history)
            n = len(sorted_scores)
            mid = n // 2

            if n % 2 == 0:
                median= (sorted_scores[mid - 1] + sorted_scores[mid]) / 2
            else:
                median = sorted_scores[mid]

            print(f'Median: {median}')