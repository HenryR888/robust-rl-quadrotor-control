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
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, sync_envs_normalization
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.evaluation import evaluate_policy

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
    
class RateEvalCallback(EvalCallback):
      '''
      Here we change the way in which the policy is checkpointed and chosen. Instead of based on total reward, we choose the policy based on best reward-per-step (total reward/total timesteps alive)
      '''

      def __init__(self, *args, **kwargs):
          super().__init__(*args, **kwargs)
          self.best_reward_rate = -np.inf

      def _on_step(self) -> bool:
          continue_training = True

          if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
              if self.model.get_vec_normalize_env() is not None:
                  try:
                      sync_envs_normalization(self.training_env, self.eval_env)
                  except AttributeError as e:
                      raise AssertionError(
                          "check the eval callback in the SB3 code"
                      ) from e

              self._is_success_buffer = []

              episode_rewards, episode_lengths = evaluate_policy(
                  self.model,
                  self.eval_env,
                  n_eval_episodes=self.n_eval_episodes,
                  render=self.render,
                  deterministic=self.deterministic,
                  return_episode_rewards=True,
                  warn=self.warn,
                  callback=self._log_success_callback,
              )

              if self.log_path is not None:
                  self.evaluations_timesteps.append(self.num_timesteps)
                  self.evaluations_results.append(episode_rewards)
                  self.evaluations_length.append(episode_lengths)

                  kwargs = {}
                  if len(self._is_success_buffer) > 0:
                      self.evaluations_successes.append(self._is_success_buffer)
                      kwargs = dict(successes=self.evaluations_successes)

                  np.savez(
                      self.log_path,
                      timesteps=self.evaluations_timesteps,
                      results=self.evaluations_results,
                      ep_lengths=self.evaluations_length,
                      **kwargs,
                  )

              mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
              mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
              self.last_mean_reward = float(mean_reward)

              # reward per step, averaged across eval episodes will be our new selection metric:
              episode_rates = np.array(episode_rewards) / np.array(episode_lengths)
              mean_reward_rate = float(np.mean(episode_rates))

              if self.verbose >= 1:
                  print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                  print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                  print(f"Reward rate: {mean_reward_rate:.3f}/step")

              self.logger.record("eval/mean_reward", float(mean_reward))
              self.logger.record("eval/mean_ep_length", mean_ep_length)
              self.logger.record("eval/mean_reward_rate", mean_reward_rate)

              if len(self._is_success_buffer) > 0:
                  success_rate = np.mean(self._is_success_buffer)
                  if self.verbose >= 1:
                      print(f"Success rate: {100 * success_rate:.2f}%")
                  self.logger.record("eval/success_rate", success_rate)

              self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
              self.logger.dump(self.num_timesteps)

              # for logging: 
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = float(mean_reward)

              if mean_reward_rate > self.best_reward_rate:
                  if self.verbose >= 1:
                      print("New best reward rate!")
                  if self.best_model_save_path is not None:
                      self.model.save(os.path.join(self.best_model_save_path, "best_model"))
                  self.best_reward_rate = mean_reward_rate
                  if self.callback_on_new_best is not None:
                      continue_training = self.callback_on_new_best.on_step()

              if self.callback is not None:
                  continue_training = continue_training and self._on_event()

          return continue_training

def train_ppo(total_timesteps: int = 20_000_000, n_envs: int =4):

    check_env(HoverEnv(), warn=True)

    # here we run n_envs in parallel to sample much for data for PPO to train on: 
    env = make_vec_env(lambda: RelativeObsWrapper(HoverEnv()), n_envs=n_envs)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    eval_env = VecNormalize(DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv())]), norm_obs=True, norm_reward=False)

    save_norm_callback = SaveNormalizeCallback(
        vec_normalize_env=env,
        save_path=os.path.join(MODEL_DIR, "best_vec_normalize.pkl")
    )

    eval_callback = RateEvalCallback(
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
        n_epochs=5, # we pass over each batch 10 times for gradient updates
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

def train_ppo_curriculum(
        # for phase2 of training we add another 20M timesteps with 1.5m plus calm conditions:
        phase2_timesteps: int = 20_000_000, 
        # and then for phase3 of training with 20M timesteps with 1.5m plus AR(1) stochastic wind conditions:
        phase3_timesteps: int = 20_000_000,
        # then we add a fourth phase of 20M timesteps with 5m plus calm conditions:
        phase4_timesteps: int = 20_000_000,
        # finally, we do a final fifth phase with 5m plus AR(1) stochastic wind conditions:
        phase5_timesteps: int = 20_000_000,
        n_envs: int = 4,
        base_model_path: str = "models/ppo/best_model",
): # after our base model finishes, we take that model, and train 20M timesteps (which is phase2 best_model). Upon completion, we take phase2_best_model and use that as the base model for phase3 to train on. This is called curriculum based training. 
    PHASE2_DIR = "models/ppo_phase2"
    PHASE3_DIR = "models/ppo_phase3"
    PHASE4_DIR = "models/ppo_phase4"
    PHASE5_DIR = "models/ppo_phase5"
    os.makedirs(PHASE2_DIR, exist_ok=True)
    os.makedirs(PHASE3_DIR, exist_ok=True)
    os.makedirs(PHASE4_DIR, exist_ok=True)
    os.makedirs(PHASE5_DIR, exist_ok=True)

    # Phase 2 of training:

    # here we run n_envs in parallel to sample much for data for PPO to train on:
    env2 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=1.5)),
        n_envs = n_envs
    )
    env2 = VecNormalize.load("models/ppo/vec_normalize.pkl", env2)
    env2.training = True
    env2.norm_reward = True
    env2.clip_obs = 10.0

    eval_env2 = VecNormalize.load(
        "models/ppo/vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=1.5))]),
    )
    eval_env2.training = False
    eval_env2.norm_reward = False

    save_norm2 = SaveNormalizeCallback(
        vec_normalize_env=env2,
        save_path = os.path.join(PHASE2_DIR, "best_vec_normalize.pkl")
    )
    eval_cb2 = RateEvalCallback(
        eval_env2,
        best_model_save_path=PHASE2_DIR,
        log_path=LOG_DIR+"_phase2",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps. 
        n_eval_episodes=10, # here we run 10 test episodes to obtain the mean reward
        deterministic=True, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net. 
        render=False,
        callback_on_new_best=save_norm2
    )

    print("Starting Phase 2...Loading...")
    model = PPO.load(base_model_path, env=env2)
    model.learn(total_timesteps=phase2_timesteps, callback=eval_cb2, reset_num_timesteps=False)
    env2.save(os.path.join(PHASE2_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE2_DIR, "final_model"))
    print(f"Phase 2 Training is complete.")

    # Phase 3:

    env3 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, wind_randomize=True, reset_radius=1.5)),
        n_envs=n_envs
    )
    env3 = VecNormalize.load("models/ppo_phase2/vec_normalize.pkl", env3)
    env3.training=True
    env3.norm_reward = True
    env3.clip_obs = 10.0
    env3.ret_rms = RunningMeanStd(shape=())

    eval_env3 = VecNormalize.load(
        "models/ppo_phase2/vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, reset_radius=1.5))])
    )
    eval_env3.training = False
    eval_env3.norm_reward = False

    save_norm3 = SaveNormalizeCallback(
        vec_normalize_env=env3,
        save_path=os.path.join(PHASE3_DIR, "best_vec_normalize.pkl")
    )
    eval_cb3 = RateEvalCallback(
        eval_env3,
        best_model_save_path=PHASE3_DIR,
        log_path=LOG_DIR + "_phase3",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=10, # here we run 10 test episodes to obtain the mean reward
        deterministic=True, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        render=False,
        callback_on_new_best=save_norm3
    )

    print("Starting Phase 3...Loading...")
    model = PPO.load(os.path.join(PHASE2_DIR, "best_model"), env=env3)
    model.learn(total_timesteps=phase3_timesteps, callback=eval_cb3, reset_num_timesteps=False)
    env3.save(os.path.join(PHASE3_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE3_DIR, "final_model"))
    print(f"Phase 3 Training is complete.")

    # Phase 4 of training:

    env4 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env4 = VecNormalize.load("models/ppo_phase3/vec_normalize.pkl", env4)
    env4.training = True
    env4.norm_reward = True
    env4.clip_obs = 10.0
    env4.ret_rms = RunningMeanStd(shape=())

    eval_env4 = VecNormalize.load(
        "models/ppo_phase3/vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env4.training = False
    eval_env4.norm_reward = False

    save_norm4 = SaveNormalizeCallback(
        vec_normalize_env=env4,
        save_path=os.path.join(PHASE4_DIR, "best_vec_normalize.pkl")
    )
    eval_cb4 = RateEvalCallback(
        eval_env4,
        best_model_save_path=PHASE4_DIR,
        log_path=LOG_DIR + "_phase4",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=10, # here we run 10 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm4
    )

    print("Starting Phase 4...Loading...")
    model = PPO.load(os.path.join(PHASE3_DIR, "best_model"), env=env4)
    model.learn(total_timesteps=phase4_timesteps, callback=eval_cb4, reset_num_timesteps=False)
    env4.save(os.path.join(PHASE4_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE4_DIR, "final_model"))
    print("Phase 4 Training is complete.")

    # Phase 5 of training:

    env5 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, wind_randomize=True, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env5 = VecNormalize.load("models/ppo_phase4/vec_normalize.pkl", env5)
    env5.training = True
    env5.norm_reward = True
    env5.clip_obs = 10.0
    env5.ret_rms = RunningMeanStd(shape=())

    eval_env5 = VecNormalize.load(
        "models/ppo_phase4/vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env5.training = False
    eval_env5.norm_reward = False

    save_norm5 = SaveNormalizeCallback(
        vec_normalize_env=env5,
        save_path=os.path.join(PHASE5_DIR, "best_vec_normalize.pkl")
    )
    eval_cb5 = RateEvalCallback(
        eval_env5,
        best_model_save_path=PHASE5_DIR,
        log_path=LOG_DIR + "_phase5",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=10, # here we run 10 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm5
    )

    print("Starting Phase 5...Loading...")
    model = PPO.load(os.path.join(PHASE4_DIR, "best_model"), env=env5)
    model.learn(total_timesteps=phase5_timesteps, callback=eval_cb5, reset_num_timesteps=False)
    env5.save(os.path.join(PHASE5_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE5_DIR, "final_model"))
    print("Phase 5 Training is complete.")

def train_ppo_curriculum_from_phase3(
        # and then for phase3 of training with 20M timesteps with 1.5m plus AR(1) stochastic wind conditions:
        phase3_timesteps: int = 20_000_000,
        # then we add a fourth phase of 20M timesteps with 5m plus calm conditions:
        phase4_timesteps: int = 20_000_000,
        # finally, we do a final fifth phase with 5m plus AR(1) stochastic wind conditions:
        phase5_timesteps: int = 20_000_000,
        n_envs: int = 4,
): # after our base model finishes, we take that model, and train 20M timesteps (which is phase2 best_model). Upon completion, we take phase2_best_model and use that as the base model for phase3 to train on. This is called curriculum based training. 
    PHASE2_DIR = "models/ppo_phase2"
    PHASE3_DIR = "models/ppo_phase3"
    PHASE4_DIR = "models/ppo_phase4"
    PHASE5_DIR = "models/ppo_phase5"
    os.makedirs(PHASE3_DIR, exist_ok=True)
    os.makedirs(PHASE4_DIR, exist_ok=True)
    os.makedirs(PHASE5_DIR, exist_ok=True)


    # Phase 3:

    env3 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, wind_randomize=True, reset_radius=1.5)),
        n_envs=n_envs
    )
    env3 = VecNormalize.load("models/ppo_phase2/best_vec_normalize.pkl", env3)
    env3.training=True
    env3.norm_reward = True
    env3.clip_obs = 10.0
    env3.ret_rms = RunningMeanStd(shape=())

    eval_env3 = VecNormalize.load(
        "models/ppo_phase2/best_vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, reset_radius=1.5))])
    )
    eval_env3.training = False
    eval_env3.norm_reward = False

    save_norm3 = SaveNormalizeCallback(
        vec_normalize_env=env3,
        save_path=os.path.join(PHASE3_DIR, "best_vec_normalize.pkl")
    )
    eval_cb3 = RateEvalCallback(
        eval_env3,
        best_model_save_path=PHASE3_DIR,
        log_path=LOG_DIR + "_phase3",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=30, # here we run 30 test episodes to obtain the mean reward
        deterministic=True, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        render=False,
        callback_on_new_best=save_norm3
    )

    print("Starting Phase 3...Loading...")
    model = PPO.load(os.path.join(PHASE2_DIR, "best_model"), env=env3, learning_rate=3e-5, ent_coef=0.001)
    model.learn(total_timesteps=phase3_timesteps, callback=eval_cb3, reset_num_timesteps=False)
    env3.save(os.path.join(PHASE3_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE3_DIR, "final_model"))
    print(f"Phase 3 Training is complete.")

    # Phase 4 of training:

    env4 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env4 = VecNormalize.load("models/ppo_phase3/best_vec_normalize.pkl", env4)
    env4.training = True
    env4.norm_reward = True
    env4.clip_obs = 10.0
    env4.ret_rms = RunningMeanStd(shape=())

    eval_env4 = VecNormalize.load(
        "models/ppo_phase3/best_vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env4.training = False
    eval_env4.norm_reward = False

    save_norm4 = SaveNormalizeCallback(
        vec_normalize_env=env4,
        save_path=os.path.join(PHASE4_DIR, "best_vec_normalize.pkl")
    )
    eval_cb4 = RateEvalCallback(
        eval_env4,
        best_model_save_path=PHASE4_DIR,
        log_path=LOG_DIR + "_phase4",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=30, # here we run 30 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm4
    )

    print("Starting Phase 4...Loading...")
    model = PPO.load(os.path.join(PHASE3_DIR, "best_model"), env=env4, learning_rate=3e-5, ent_coef=0.001)
    model.learn(total_timesteps=phase4_timesteps, callback=eval_cb4, reset_num_timesteps=False)
    env4.save(os.path.join(PHASE4_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE4_DIR, "final_model"))
    print("Phase 4 Training is complete.")

    # Phase 5 of training:

    env5 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, wind_randomize=True, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env5 = VecNormalize.load("models/ppo_phase4/best_vec_normalize.pkl", env5)
    env5.training = True
    env5.norm_reward = True
    env5.clip_obs = 10.0
    env5.ret_rms = RunningMeanStd(shape=())

    eval_env5 = VecNormalize.load(
        "models/ppo_phase4/best_vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env5.training = False
    eval_env5.norm_reward = False

    save_norm5 = SaveNormalizeCallback(
        vec_normalize_env=env5,
        save_path=os.path.join(PHASE5_DIR, "best_vec_normalize.pkl")
    )
    eval_cb5 = RateEvalCallback(
        eval_env5,
        best_model_save_path=PHASE5_DIR,
        log_path=LOG_DIR + "_phase5",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=30, # here we run 30 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm5
    )

    print("Starting Phase 5...Loading...")
    model = PPO.load(os.path.join(PHASE4_DIR, "best_model"), env=env5, learning_rate=3e-5, ent_coef=0.001)
    model.learn(total_timesteps=phase5_timesteps, callback=eval_cb5, reset_num_timesteps=False)
    env5.save(os.path.join(PHASE5_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE5_DIR, "final_model"))
    print("Phase 5 Training is complete.")

def train_ppo_curriculum_from_phase4(
        # then we add a fourth phase of 20M timesteps with 5m plus calm conditions:
        phase4_timesteps: int = 20_000_000,
        # finally, we do a final fifth phase with 5m plus AR(1) stochastic wind conditions:
        phase5_timesteps: int = 20_000_000,
        n_envs: int = 4,
): # after our base model finishes, we take that model, and train 20M timesteps (which is phase2 best_model). Upon completion, we take phase2_best_model and use that as the base model for phase3 to train on. This is called curriculum based training. 
    PHASE2_DIR = "models/ppo_phase2"
    PHASE3_DIR = "models/ppo_phase3"
    PHASE4_DIR = "models/ppo_phase4"
    PHASE5_DIR = "models/ppo_phase5"
    os.makedirs(PHASE3_DIR, exist_ok=True)
    os.makedirs(PHASE4_DIR, exist_ok=True)
    os.makedirs(PHASE5_DIR, exist_ok=True)

    # Phase 4 of training:

    env4 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env4 = VecNormalize.load("models/ppo_phase3/best_vec_normalize.pkl", env4)
    env4.training = True
    env4.norm_reward = True
    env4.clip_obs = 10.0
    env4.ret_rms = RunningMeanStd(shape=())

    eval_env4 = VecNormalize.load(
        "models/ppo_phase3/best_vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=0.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env4.training = False
    eval_env4.norm_reward = False

    save_norm4 = SaveNormalizeCallback(
        vec_normalize_env=env4,
        save_path=os.path.join(PHASE4_DIR, "best_vec_normalize.pkl")
    )
    eval_cb4 = RateEvalCallback(
        eval_env4,
        best_model_save_path=PHASE4_DIR,
        log_path=LOG_DIR + "_phase4",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=30, # here we run 30 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm4
    )

    print("Starting Phase 4...Loading...")
    model = PPO.load(os.path.join(PHASE3_DIR, "best_model"), env=env4, learning_rate=3e-5, ent_coef=0.001)
    model.learn(total_timesteps=phase4_timesteps, callback=eval_cb4, reset_num_timesteps=False)
    env4.save(os.path.join(PHASE4_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE4_DIR, "final_model"))
    print("Phase 4 Training is complete.")

    # Phase 5 of training:

    env5 = make_vec_env(
        lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, wind_randomize=True, reset_radius=5.0, reset_sphere=True)),
        n_envs=n_envs
    )
    env5 = VecNormalize.load("models/ppo_phase4/best_vec_normalize.pkl", env5)
    env5.training = True
    env5.norm_reward = True
    env5.clip_obs = 10.0
    env5.ret_rms = RunningMeanStd(shape=())

    eval_env5 = VecNormalize.load(
        "models/ppo_phase4/best_vec_normalize.pkl",
        DummyVecEnv([lambda: RelativeObsWrapper(HoverEnv(wind_magnitude=2.0, reset_radius=5.0, reset_sphere=True))])
    )
    eval_env5.training = False
    eval_env5.norm_reward = False

    save_norm5 = SaveNormalizeCallback(
        vec_normalize_env=env5,
        save_path=os.path.join(PHASE5_DIR, "best_vec_normalize.pkl")
    )
    eval_cb5 = RateEvalCallback(
        eval_env5,
        best_model_save_path=PHASE5_DIR,
        log_path=LOG_DIR + "_phase5",
        eval_freq=max(10_000 // n_envs, 1), # we run an evaluation to update the model every 10000 time steps.
        n_eval_episodes=30, # here we run 30 test episodes to obtain the mean reward
        deterministic=True,render=False, # we remove the stochastic distribution (Gaussian in this case) when producing the output of our Net.
        callback_on_new_best=save_norm5
    )

    print("Starting Phase 5...Loading...")
    model = PPO.load(os.path.join(PHASE4_DIR, "best_model"), env=env5, learning_rate=3e-5, ent_coef=0.001)
    model.learn(total_timesteps=phase5_timesteps, callback=eval_cb5, reset_num_timesteps=False)
    env5.save(os.path.join(PHASE5_DIR, "vec_normalize.pkl"))
    model.save(os.path.join(PHASE5_DIR, "final_model"))
    print("Phase 5 Training is complete.")


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
