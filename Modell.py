from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.ppo_mask import MaskablePPO
from sb3_contrib.common.maskable.utils import get_action_masks
from stable_baselines3.common.vec_env import SubprocVecEnv
from GanzSchoenCleverEnv import *
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
from typing import cast
import torch.nn as nn


# learning_process for the model
def model_learn(n_envs=32, name="maskableppo_ganzschoenclever", net_arch=None, activation_fn=nn.ReLU, gamma=1,
                learning_rate=0.0003*2, ent_coef=0.05, clip_range=0.2, verbose=1, n_steps=int(2048 / 32), n_epochs=5,
                batch_size=int(2048 / 16), total_timesteps=1000000, prediction_ent_coef=0, prediction_gamma=1):
    envs = _init_envs(n_envs)
    policy_kwargs = dict(net_arch=net_arch, activation_fn=activation_fn)
    model = MaskablePPO(MaskableActorCriticPolicy, envs, gamma=gamma, learning_rate=learning_rate,
                        policy_kwargs=policy_kwargs,
                        ent_coef=ent_coef, clip_range=clip_range, verbose=verbose, n_steps=n_steps, n_epochs=n_epochs,
                        batch_size=batch_size)

    model.learn(total_timesteps=total_timesteps)
    model.ent_coef = prediction_ent_coef
    model.gamma = prediction_gamma
    model.save(name)


# making predictions with the model
def model_predict(n_steps=200, model_name="maskableppo_ganzschoenclever", n_envs=1, render=False, render_mode="human"):
    model = MaskablePPO.load(model_name)
    envs, scores, score_history, fails, fail_history = \
        _init_envs(n_envs, n_envs, scores=True, fails=True, render_mode=render_mode)
    obs = envs.reset()
    for i in range(n_steps):
        action_masks = get_action_masks(envs)
        action, _states = model.predict(obs, action_masks=action_masks)
        obs, rewards, dones, info = envs.step(action)
        make_score_entries(rewards, scores, number_of_entries=n_envs)
        make_fail_entries(rewards, fails, number_of_entries=n_envs)
        make_score_history_entry(dones, scores, score_history, number_of_entries=n_envs)
        make_fail_history_entry(dones, fails, fail_history, number_of_entries=n_envs)
        if render is True:
            envs.render()

    plot_history(score_history, "score")
    plot_history(fail_history, "fails")


# making a fail entry
def make_fail_entries(rewards, fails, number_of_entries=4):
    for i in range(number_of_entries):
        if rewards[i] < 0:
            fails[i] += 1


# making a score entry
def make_score_entries(rewards, scores, number_of_entries=4):
    for i in range(number_of_entries):
        if rewards[i] > 0:
            scores[i] += rewards[i]


# making a fail_history entry
def make_fail_history_entry(dones, fails, fail_history, number_of_entries=4):
    for i, done in enumerate(dones):
        if done and i < number_of_entries:
            fail_history[i].append(fails[i])
            fails[i] = 0


# making a score_history entry
def make_score_history_entry(dones, scores, score_history, number_of_entries=4):
    for i, done in enumerate(dones):
        if done and i < number_of_entries:
            score_history[i].append(scores[i])
            scores[i] = 0


# plotting a history for visualization
def plot_history(history, name):
    for i, history_entry in enumerate(history):
        plt.figure()
        plt.plot(history_entry)
        plt.title(f"Environment {i + 1} " + name)
        plt.xlabel("Episode")
        plt.ylabel("Value")
        plt.show()


# returning action mask for a specific environment
def mask_fn(env_clever: gym.Env) -> np.ndarray:
    env_clever = cast(GanzSchonCleverEnv, env_clever)
    return env_clever.valid_action_mask()


# initializing environments
def _init_envs(n_envs=1, number_of_entries=None, scores=False, fails=False, render_mode="human"):
    def _init():
        env_make = GanzSchonCleverEnv(render_mode=render_mode)
        env_make = ActionMasker(env_make, mask_fn)
        return env_make

    envs_make = SubprocVecEnv([_init for _ in range(n_envs)])
    scores_make = None
    score_history_make = None
    fails_make = None
    fail_history_make = None

    if scores is True:
        scores_make = np.zeros(number_of_entries)
        score_history_make = [[] for _ in range(number_of_entries)]
    if fails is True:
        fails_make = np.zeros(number_of_entries)
        fail_history_make = [[] for _ in range(number_of_entries)]

    if scores is False and fails is False:
        return envs_make
    elif scores is True and fails is False:
        return envs_make, scores_make, score_history_make
    elif scores is False and fails is True:
        return envs_make, fails_make, fail_history_make
    elif scores is True and fails is True:
        return envs_make, scores_make, score_history_make, fails_make, fail_history_make
    