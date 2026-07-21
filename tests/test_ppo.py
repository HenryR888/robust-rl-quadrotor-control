'''
Here we test PPO controller. We validate tha environment and complete a short training run as a smoke test.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env

from envs.hover_env import HoverEnv
from controllers.ppo import train_ppo, train_ppo_curriculum, train_ppo_curriculum_from_phase3, train_ppo_curriculum_from_phase4, PPOController, RelativeObsWrapper, MODEL_DIR


print("=== 1. Gymnasium env. Check ===")
check_env(HoverEnv(), warn=True)
print("HoverEnv passed check_env.\n")

# here we check to see that the relative observation wrapper works correctly and sets desired target to [0,0,0]: 
wrapped = RelativeObsWrapper(HoverEnv(target=np.array([0.0, 0.0, 1.0])))
obs, _ = wrapped.reset()
raw_env= wrapped.env
expected_relative = raw_env.state[0:3] - raw_env.target
assert np.allclose(obs[0:3], expected_relative), f"Expected {expected_relative} got {obs[0:3]}"
print(f"Relative pos at reset: {obs[0:3]}  (expected {expected_relative})")
print("RelativeObsWrapper passed.\n")

# we run 20M timestep simulation to test that the environment and training pipeline works correctly without crashing: 
#train_ppo(total_timesteps=20_000_000, n_envs=4)
#train_ppo_curriculum(phase2_timesteps=20_000_000, phase3_timesteps=20_000_000, phase4_timesteps=20_000_000, phase5_timesteps=20_000_000, n_envs=4)
#train_ppo_curriculum_from_phase3(phase3_timesteps=20_000_000, phase4_timesteps=20_000_000, phase5_timesteps=20_000_000, n_envs=4)
train_ppo_curriculum_from_phase4(phase4_timesteps=20_000_000, phase5_timesteps=20_000_000, n_envs=4)


print("=== 2. Phase 1 Evaluation (no wind, near target) ===")
controller1 = PPOController(model_path="models/ppo/best_model", norm_path="models/ppo/best_vec_normalize.pkl")
env1 = HoverEnv(target=np.array([0.0, 0.0, 1.0]))
obs, _ = env1.reset(seed=0)
states1, rewards1 = [obs.copy()], []
for step in range(env1.max_steps):
    action = controller1.compute_action(obs, env1.target, env1.dt)
    obs, reward, terminated, truncated, _ = env1.step(action)
    states1.append(obs.copy())
    rewards1.append(reward)
    if terminated or truncated:
        print(f"Phase 1: ended at step {step+1} (terminated={terminated})")
        break
else: 
    print(f"Phase 1: completed all {env1.max_steps} steps")

print("=== 3. Phase 2 Evaluation (calm, start 1.5m from target) ===")
controller2 = PPOController(model_path="models/ppo_phase2/best_model", norm_path="models/ppo_phase2/best_vec_normalize.pkl")
env2 = HoverEnv(target=np.array([0.0, 0.0, 1.0]), wind_magnitude=0.0)
obs, _ = env2.reset(seed=0)
env2.state[0:3] = np.array([1.5, 0.0, 1.0])
obs = env2.state.copy()
states2, rewards2 = [obs.copy()], []
for step in range(env2.max_steps):
    action = controller2.compute_action(obs, env2.target, env2.dt)
    obs, reward, terminated, truncated, _ = env2.step(action)
    states2.append(obs.copy())
    rewards2.append(reward)
    if terminated or truncated:
        print(f"Phase 2: ended at step {step+1} (terminated={terminated})")
        break
else:
    print(f"Phase 2: completed all {env2.max_steps} steps")

print("=== 4. Phase 3 Evaluation (wind=2N, start 1.5m from target) ===")
controller3 = PPOController(model_path="models/ppo_phase3/best_model", norm_path="models/ppo_phase3/best_vec_normalize.pkl")
env3 = HoverEnv(target=np.array([0.0, 0.0, 1.0]), wind_magnitude=2.0)
obs, _ = env3.reset(seed=0)
env3.state[0:3] = np.array([1.5, 0.0, 1.0])
obs = env3.state.copy()
states3, rewards3 = [obs.copy()], []
for step in range(env3.max_steps):
    action = controller3.compute_action(obs, env3.target, env3.dt)
    obs, reward, terminated, truncated, _ = env3.step(action)
    states3.append(obs.copy())
    rewards3.append(reward)
    if terminated or truncated:
        print(f"Phase 3: ended at step {step+1} (terminated={terminated})")
        break
else: 
    print(f"Phase 3: completed all {env3.max_steps} steps")

print("=== 5. Phase 4 Evaluation (calm, start 5m from target) ===")
controller4 = PPOController(model_path="models/ppo_phase4/best_model", norm_path="models/ppo_phase4/best_vec_normalize.pkl")
env4 = HoverEnv(target=np.array([0.0, 0.0, 1.0]), wind_magnitude=0.0)
obs, _ = env4.reset(seed=0)
env4.state[0:3] = np.array([5.0, 0.0, 1.0])
obs = env4.state.copy()
states4, rewards4 = [obs.copy()], []
for step in range(env4.max_steps):
    action = controller4.compute_action(obs, env4.target, env4.dt)
    obs, reward, terminated, truncated, _ = env4.step(action)
    states4.append(obs.copy())
    rewards4.append(reward)
    if terminated or truncated:
        print(f"Phase 4: ended at step {step+1} (terminated={terminated})")
        break
else: 
    print(f"Phase 4: completed all {env4.max_steps} steps")

print("=== 6. Phase 5 Evaluation (wind=2N, start 5m from target) ===")
controller5 = PPOController(model_path="models/ppo_phase5/best_model", norm_path="models/ppo_phase5/best_vec_normalize.pkl")
env5 = HoverEnv(target=np.array([0.0, 0.0, 1.0]), wind_magnitude=2.0)
obs, _ = env5.reset(seed=0)
env5.state[0:3] = np.array([5.0, 0.0, 1.0])
obs = env5.state.copy()
states5, rewards5 = [obs.copy()], []
for step in range(env5.max_steps):
    action = controller5.compute_action(obs, env5.target, env5.dt)
    obs, reward, terminated, truncated, _ = env5.step(action)
    states5.append(obs.copy())
    rewards5.append(reward)
    if terminated or truncated:
        print(f"Phase 5: ended at step {step+1} (terminated={terminated})")
        break
else:
    print(f"Phase 5: completed all {env5.max_steps} steps")





def plot_phase(states, rewards, env, title):
    states = np.array(states)
    t = np.arange(len(states)) * env.dt
    pos_error = np.linalg.norm(states[:, 0:3] -env.target, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    ax = axes[0, 0]
    for i, label in enumerate(['x', 'y', 'z']):
        ax.plot(t, states[:, i], label=label)
    ax.axhline(env.target[2], color='r', linestyle='--', linewidth=1, label='z target')
    ax.set_ylabel('Position (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Position')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[0, 1]
    ax.plot(t, pos_error, color='steelblue')
    ax.set_ylabel('||p - p*|| (m)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Position Error')
    ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[1, 0]
    for i, label in [(6, r'$\phi$ roll'), (7, r'$\theta$ pitch'), (8, r'$\psi$ yaw')]:
        ax.plot(t, states[:, i], label=label)
    ax.axhline(0.0, color='r', linestyle='--', linewidth=1)
    ax.set_ylabel('Euler Angles (rad)')
    ax.set_xlabel('Time (s)')
    ax.set_title('Roll, Pitch, Yaw')
    ax.legend(fontsize=8)   
    ax.grid(True, linestyle=':', alpha=0.6)

    ax = axes[1, 1]
    ax.plot(t[1:], rewards, color='steelblue')
    ax.set_ylabel('Reward')
    ax.set_xlabel('Time (s)')
    ax.set_title('Per-step Reward')
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    plt.show()

plot_phase(states1, rewards1, env1, "Phase 1: No Wind, Near Target")
plot_phase(states2, rewards2, env2, "Phase 2: Calm, Start 1.5m from Target")
plot_phase(states3, rewards3, env3, "Phase 3: Wind=2N, Start 1.5m from Target")
plot_phase(states4, rewards4, env4, "Phase 4: Calm, Start 5m from Target")
plot_phase(states5, rewards5, env5, "Phase 5: Wind=2N, Start 5m from Target")