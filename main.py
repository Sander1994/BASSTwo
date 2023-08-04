from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from GanzSchoenCleverEnv import GanzSchonCleverEnv
import matplotlib.pyplot as plt


def train_and_test_model():
    n_envs = 8  # Number of environments to create

    # Function to create a single environment, this will be called n_envs times
    def make_env():
        def _init():
            return GanzSchonCleverEnv()
        return _init

    # Create the vectorized environment
    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=2500000)
    #model.save("ppo_ganzschoenclever")

    #del model

    model = PPO.load("ppo_ganzschoenclever")

    obs = env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        if dones.any():
            obs = env.reset()  # Note: with multiple environments, you reset only the ones that are done


def main():
    # Other parts of your code

    # At some point you decide to start the training and testing of the model
    train_and_test_model()

    # More of your code


if __name__ == "__main__":
    main()
