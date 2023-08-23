from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from GanzSchoenCleverEnv import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from typing import cast


def train_and_test_model():
    n_envs = 32

    def make_env():
        def _init():
            build_env = GanzSchonCleverEnv()
            return ActionMasker(build_env, mask_fn)

        return _init

    env = SubprocVecEnv([make_env() for _ in range(n_envs)])
    scores = np.zeros(n_envs)
    scores_history = [[] for _ in range(4)]
    fails = np.zeros(n_envs)
    fails_history = [[] for _ in range(4)]

    policy_kwargs = dict(net_arch=[256, 256, 256])
    model = MaskablePPO(MaskableActorCriticPolicy, env, gamma=0.9, learning_rate=0.0003*2,
                        policy_kwargs=policy_kwargs,
                        ent_coef=0.01, clip_range=0.2, verbose=1, n_steps=int(2048 / 32), n_epochs=10,
                        batch_size=int(2048 / 16))

    model.learn(total_timesteps=1000000)
    model.ent_coef = 0
    model.save("maskableppo_ganzschoenclever")

    model = MaskablePPO.load("maskableppo_ganzschoenclever")
    model.gamma = 0

    obs = env.reset()
    j = 0
    while j < 200:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        j += 1

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
                if i < 4:
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


def mask_fn(envclever: gym.Env) -> np.ndarray:
    env_clever = cast(GanzSchonCleverEnv, envclever)
    return env_clever.valid_action_mask_value
