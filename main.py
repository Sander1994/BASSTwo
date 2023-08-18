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

    scores = np.zeros(n_envs)
    scores_history = [[] for _ in range(n_envs)]
    policy_kwargs = dict(net_arch=[96, 96, 96])

    model = PPO("MlpPolicy", env, gamma=1, learning_rate=0.0001, policy_kwargs=policy_kwargs, ent_coef=0.01, verbose=1)
    model.learn(total_timesteps=25000000)
    model.save("ppo_ganzschoenclever")

    model = PPO.load("ppo_ganzschoenclever")

    obs = env.reset()
    j = 0
    while j < 200:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        j += 1
        scores_old = scores

        for i in range(4):
            if rewards[i] > 9:
                scores[i] += rewards[i]

        for i in range(4):
            # if scores[i] > scores_old[i] | scores == 0:
            #     print("True in Step " + str(j))
            # else:
            #     print("False in Step " + str(j))
            print("Env:" + str(i) + " Points:" + str(scores[i]) + " in J:" + str(j))

        for i, done in enumerate(dones):
            if done:
                scores_history[i].append(scores[i])  # Store the score for this episode
                scores[i] = 0  # Reset the score
                obs[i] = env.reset()[i]  # Reset the done environment

    for i, score_history in enumerate(scores_history):
        plt.figure()
        plt.plot(score_history)
        plt.title(f'Environment {i + 1} Score History')
        plt.xlabel('Episode')
        plt.ylabel('Score')
        plt.show()


def main():

    train_and_test_model()


if __name__ == "__main__":
    main()
