import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from GanzSchoenCleverEnv import GanzSchonCleverEnv

# Instantiate your environment
env = GanzSchonCleverEnv(rounds=10)

# Use the function to check your environment
check_env(env)
