'''
Here we implement the Proximal Policy Optimisation (PPO) controller for the quadrotor. We train PPO via the Stable-Baseline3 library.
'''

import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.hover_env import HoverEnv

# we create two directories: one is for the model directory, for which when we run the EvalCallback, we can store the best model which is found by pausing the rollout and evaluating
# the mean reward over a specific number of test episodes and then saving policy which produced the highest mean reward. We also keep track of the logs within the log directory which we can subsequently use.
# Finally, we use the tensorboard logger to plot all of our results from pytorch
MODEL_DIR = "models/ppo"
LOG_DIR = "logs/ppo"
TENSORBOARD_LOG_DIR = "tensorboard/ppo"

# we create a wrapper to be able to generalise the target that we want the quadrotor to reach: 
class RelativeObsWrapper(gym.ObservationWrapper):
    def observation(self, obs):
        obs = obs.copy()
        obs[0:3] = obs[0:3] - self.env.target
        return obs
    
class SaveNormalizeCallback(BaseCallback):
    def __init__(self, vec_normalize_env, save_path, verbose=0):
        super().__init__(verbose)
        self.vec_normalize_env = vec_normalize_env
        self.save_path = save_path

    def _on_step(self) -> bool:
        self.vec_normalize_env.save(self.save_path)
        return True

def train_ppo(total_timesteps: int = 10_000_000, n_envs: int =4):

    check_env(HoverEnv(), warn=True)

    # here we run n_envs in parallel to sample much for data for PPO to train on: 
    env = make_vec_env(lambda: RelativeObsWrapper(HoverEnv()), n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv())]), norm_obs=True, norm_reward=False)

    save_norm_callback = SaveNormalizeCallback(
        vec_normalize_env=env,
        save_path=os.path.join(MODEL_DIR, "best_vec_normalize.pkl")
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps. 
        n_eval_episodes=10, # here we run 10 test episodes to obtain the mean reward
        deterministic=True, # we remove the noise during evaluation
        render=False,
        callback_on_new_best=save_norm_callback
    )

    model = PPO(
        "MlpPolicy", # we use a standard multi-layer perceptron as the architecture for our NN, since we have a simple 12-D vector as input and simple 4-D vector as output
        env, # here we choose the environment from which we want our 'agent' (drone in this case) to sample experience from
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
        policy_kwargs={"net_arch": [128,128]}, # we use 2x128 hidden layers, for actor and 2 x128 hidden layer size for critic
        learning_rate=1e-4, 
        n_steps=2048,
        batch_size=64, # we split the samples into mini-batches of 64 for gradient update
        n_epochs=10, # we pass over each batch 10 times for gradient updates
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    env.save(os.path.join(MODEL_DIR, "vec_normalize.pkl"))

    os.makedirs(MODEL_DIR, exist_ok=True)
    model.save(os.path.join(MODEL_DIR,"final_model"))
    print(f"Training is complete.")
    return model

class PPOController:

    def __init__(self, model_path: str, norm_path: str):
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        self.model = PPO.load(model_path)
        vec_env = DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv())])
        self.env = VecNormalize.load(norm_path, vec_env)
        self.env.training=False
        self.env.norm_reward = False

    def reset(self):
        pass

    def compute_action(self, obs: np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        """
        Here our obs is the full 12-state vector given by: [x,y,z,vx,vy,vz,phi,theta,psi,wx,wy,wz]
        - target is: desired position [x*, y*, z*]
        - dt is not used here but we need it to fit into our hover_env structure
        We return [T, tau_x, tau_y, tau_z]
        """
        relative_obs = obs.copy()
        relative_obs[0:3] = obs[0:3] - target
        obs_norm = self.env.normalize_obs(relative_obs)
        action, _ = self.model.predict(obs_norm, deterministic=True) # here we take in the relative observation for our observation and pass that into our actor network, and output our [T,tau_x,tau_y,tau_z] action 
        return action
