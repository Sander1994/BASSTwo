from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from GanzSchoenCleverEnv import GanzSchonCleverEnv
import matplotlib.pyplot as plt
import numpy as np


def train_and_test_model():
    n_envs = 6

    def make_env():
        def _init():
            return GanzSchonCleverEnv()
        return _init

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])

    scores = np.zeros(n_envs)
    scores_history = [[] for _ in range(n_envs)]
    fails = np.zeros(n_envs)
    fails_history = [[] for _ in range(n_envs)]
    policy_kwargs = dict(net_arch=[64, 64, 64, 64])

    model = PPO("MlpPolicy", env, gamma=1, learning_rate=0.0003*4, policy_kwargs=policy_kwargs,
                ent_coef=0.01, clip_range=0.2, verbose=1, n_steps=int(2048*8), n_epochs=42,
                batch_size=int(64*4), device="cpu")
    model.learn(total_timesteps=1000000)
    model.save("ppo_ganzschoenclever")

    model = PPO.load("ppo_ganzschoenclever")

    obs = env.reset()
    j = 0
    while j < 200:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        j += 1
        scores_old = scores

        for i in range(n_envs):
            if rewards[i] > 9:
                scores[i] += rewards[i]
            if rewards[i] == -15:
                fails[i] += 1

        for i in range(n_envs):
            # if scores[i] > scores_old[i] | scores == 0:
            #     print("True in Step " + str(j))
            # else:
            #     print("False in Step " + str(j))
            print("Env:" + str(i) + " Points:" + str(scores[i]) + " in J:" + str(j))

        for i, done in enumerate(dones):
            if done:
                scores_history[i].append(scores[i])
                fails_history[i].append(fails[i])
                scores[i] = 0
                fails[i] = 0
                obs[i] = env.reset()[i]

    # for i, score_history in enumerate(scores_history):
    #     plt.figure()
    #     plt.plot(score_history)
    #     plt.title(f'Environment {i + 1} Score History')
    #     plt.xlabel('Episode')
    #     plt.ylabel('Score')
    #     plt.show()

    for i, fail_history in enumerate(fails_history):
        plt.figure()
        plt.plot(fail_history)
        plt.title(f'Environment {i + 1} Fail History')
        plt.xlabel('Episode')
        plt.ylabel('Fails')
        plt.show()


def main():

    train_and_test_model()


if __name__ == "__main__":
    main()
