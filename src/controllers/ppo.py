'''
Here we implement the Proximal Policy Optimisation (PPO) controller for the quadrotor. We train PPO via the Stable-Baseline3 library.
'''

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_checker import check_env

from envs.hover_env import HoverEnv

def train_ppo(total_timesteps: int = 1_000_000, n_envs: int =4):

    check_env(HoverEnv(), warn=True)

    # here we run n_envs in parallel to sample much for data for PPO to train on: 
    env = make_vec_env(HoverEnv, n_envs=n_envs)

