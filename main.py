from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from GanzSchoenCleverEnv import GanzSchonCleverEnv
import matplotlib.pyplot as plt
import numpy as np


def train_and_test_model():
    n_envs = 4

    def make_env():
        def _init():
            return GanzSchonCleverEnv()
        return _init

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    # Initialize the data structure for scores
    scores = np.zeros(n_envs)
    scores_history = [[] for _ in range(n_envs)]

    #model = PPO("MlpPolicy", env, verbose=1)
    #model.learn(total_timesteps=2500000)
    #model.save("ppo_ganzschoenclever")

    #del model

    model = PPO.load("ppo_ganzschoenclever")

    obs = env.reset()
    j = 0
    while j < 50000:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        j += 1

        # Update the scores
        scores += rewards
        for i, done in enumerate(dones):
            if done:
                scores_history[i].append(scores[i])  # Store the score for this episode
                scores[i] = 0  # Reset the score
                obs[i] = env.reset()[i]  # Reset the done environment

    # Plotting the scores
    for i, score_history in enumerate(scores_history):
        plt.figure()
        plt.plot(score_history)
        plt.title(f'Environment {i + 1} Score History')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.show()


def main():
    # Other parts of your code

    # At some point you decide to start the training and testing of the model
    train_and_test_model()

    # More of your code


if __name__ == "__main__":
    main()
