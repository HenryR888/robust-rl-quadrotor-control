'''
Here we test PPO controller. We validate tha environment and complete a short training run as a smoke test.
'''
import os
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.env_checker import check_env

from envs.hover_env import HoverEnv
from controllers.ppo import train_ppo, PPOController, RelativeObsWrapper, MODEL_DIR


print("=== 1. Gymnasium env. Check ===")
check_env(HoverEnv(), warn=True)
print("HoverEnv passed check_env.\n")

# here we check to see that the relative observation wrapper works correctly and sets desired target to [0,0,0]: 
wrapped = RelativeObsWrapper(HoverEnv(target=np.array([0.0, 0.0, 1.0])))
obs, _ = wrapped.reset()
assert np.allclose(obs[0:3], 0.0), f"Expected [0,0,0] got {obs[0:3]}"
print(f"Relative pos at reset: {obs[0:3]}  (expected [0, 0, 0])")
print("RelativeObsWrapper passed.\n")

# we run a quick 50K timestep simulation to test that the environment and training pipeline works correctly without crashing: 
train_ppo(total_timesteps=1_000_000, n_envs=4)


print("=== 4. PPOController evaluation ===")
best = f"{MODEL_DIR}/best_model"
final = f"{MODEL_DIR}/final_model"
model_path = best if os.path.exists(best + ".zip") else final
print(f"  Loading model from: {model_path}.zip")


controller = PPOController(model_path=model_path, norm_path=f"{MODEL_DIR}/vec_normalize.pkl")

# initialise environment from HoverEnv: 
env = HoverEnv(target=np.array([0.0, 0.0, 1.0]))
obs, _ = env.reset()

states = [obs.copy()]
rewards = []

for step in range(env.max_steps):
    action = controller.compute_action(obs, env.target, env.dt)
    obs, reward, terminated, truncated, _ = env.step(action)
    states.append(obs.copy())
    rewards.append(reward)
    if terminated or truncated:
        print(f"Episode ended at step {step + 1}: terminated={terminated}, truncated={truncated}")
        break
else:
    print(f"Completed all {env.max_steps} steps without termination.")

states = np.array(states)
t = np.arange(len(states)) * env.dt
pos_error = np.linalg.norm(states[:, 0:3] - env.target, axis=1)

print(f"Final Position: {states[-1, 0:3]}")
print(f"Final Position error: {pos_error[-1]:.4f} m")
print(f"Mean Position error for last 100 steps: {np.mean(pos_error[-100:]):.4f} m")
print(f"Total reward: {sum(rewards):.2f}")

# plots: 

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("PPO Controller: Hover Test (50k timestep training)", fontsize=13, fontweight='bold')
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
for i, label in [( 6, r'$\phi$ roll'), (7, r'$\theta$ pitch'), (8, r'$\psi$ yaw')]:
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


