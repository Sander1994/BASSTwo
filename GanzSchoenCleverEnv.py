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
        self.rewards = {'row': [10, 14, 16, 20], 'col': [10, 14, 16, 20]}
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.score_history = []
        self.last_dice = None

        self.action_space = spaces.Discrete(17)
        self.observation_space = spaces.Box(low=0, high=6, shape=(19,), dtype=np.int32)

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
                if reward == 0:
                    reward += 1
            else:
                terminated = True  # end episode if invalid field action is taken
                if reward == 0:
                    reward -= 1
                return self._get_obs(), reward, terminated, truncated, info

        else:
            if self.extra_pick:
                if self.last_dice:
                    found_match = False
                    for i in range(4):
                        for j in range(4):
                            if self.yellow_field[i][j] in self.last_dice and self.yellow_field[i][j] != 0:
                                self.yellow_field[i][j] = 0
                                reward += self.check_rewards()
                                self.extra_pick = False
                                found_match = True
                                break
                        if found_match:
                            break
                    if not found_match:
                        if reward == 0:
                            reward -= 1
                        terminated = True  # end episode if invalid extra pick action is taken
                        return self._get_obs(), reward, terminated, truncated, info
                else:
                    if reward == 0:
                        reward -= 1
                    terminated = True  # end episode if invalid extra pick action is taken
                    return self._get_obs(), reward, terminated, truncated, info

        # Save the current dice values for the extra pick in the next turn
        self.last_dice = self.dice

        # Reroll dice for the next turn after processing both actions
        self.dice = self.roll_dice()

        # Check if all rounds are played
        self.rounds -= 1
        self.score += reward
        if self.rounds == 0:
            terminated = True

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, **kwargs):
        self.score_history.append(self.score)
        self.yellow_field = [[3, 6, 5, 0], [2, 1, 0, 5], [1, 0, 2, 4], [0, 3, 4, 6]]
        self.reward_flags = {'row': [False, False, False, False], 'col': [False, False, False, False]}
        self.extra_pick = False
        self.score = 0
        self.rounds = self.initial_rounds
        self.dice = self.roll_dice()
        info = {}

        return self._get_obs(), info

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
        obs = np.concatenate((yellow_field_array, dice_array, [self.extra_pick]), axis=None)
        return obs
