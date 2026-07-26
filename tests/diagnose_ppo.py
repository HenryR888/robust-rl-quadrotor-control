import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv
from controllers.ppo import PPOController
from compare import run_episode, _set_ic, SCENARIOS, TARGET

PHASE5_DIR = "models/ppo_phase5"


def find_crashing_seeds(scenario_name: str, n_seeds: int = 20):
    config = SCENARIOS[scenario_name]
    controller = PPOController(
        model_path=f"{PHASE5_DIR}/best_model",
        norm_path=f"{PHASE5_DIR}/best_vec_normalize.pkl",
    )
    crashed = []
    for seed in range(n_seeds):
        env = HoverEnv(target=TARGET, **config["env_kwargs"])
        result = run_episode(controller, env, seed, config["ic"], config["approach_speed"])
        if result.crashed:
            crashed.append((seed, result.episode_length))
    print(f"{scenario_name}: crashed (seed, episode_length) = {crashed}")
    return crashed


def _run_episode_with_actions(scenario_name: str, seed: int, sim_time: float = 50.0):
    config = SCENARIOS[scenario_name]
    env = HoverEnv(target=TARGET, **config["env_kwargs"])
    controller = PPOController(
          model_path=f"{PHASE5_DIR}/best_model",
          norm_path=f"{PHASE5_DIR}/best_vec_normalize.pkl",
      )

    obs, _ = env.reset(seed=seed)
    if config["ic"] != "local":
        obs = _set_ic(env, config["ic"])
    if config["approach_speed"] > 0.0:
        azimuth = env.np_random.uniform(0, 2 * np.pi)
        elevation = env.np_random.uniform(-np.pi / 4, np.pi / 4)
        env.state[3] = config["approach_speed"] * np.cos(elevation) * np.cos(azimuth)
        env.state[4] = config["approach_speed"] * np.cos(elevation) * np.sin(azimuth)
        env.state[5] = config["approach_speed"] * np.sin(elevation)
        obs = env.state.copy()
    controller.reset()

    n_steps = int(sim_time / env.dt)
    states, actions = [], []
    terminated_early = False
    for _ in range(n_steps):
        action = controller.compute_action(obs, TARGET, env.dt)
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        if terminated or truncated:
              terminated_early = terminated
              break

    return np.array(states), np.array(actions), terminated_early


def _plot(states: np.ndarray, actions: np.ndarray, dt: float, title: str):
    t = np.arange(len(states)) * dt
    fig, axes = plt.subplots(4, 1, figsize=(10, 12))
    fig.suptitle(title)

    axes[0].plot(t, states[:, 0], label='x')
    axes[0].plot(t, states[:, 1], label='y')
    axes[0].plot(t, states[:, 2], label='z')
    axes[0].axhline(TARGET[2], color='k', linestyle='--', linewidth=0.8, label='z target')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, np.degrees(states[:, 6]), label='phi (roll)')
    axes[1].plot(t, np.degrees(states[:, 7]), label='theta (pitch)')
    axes[1].plot(t, np.degrees(states[:, 8]), label='psi (yaw)')
    axes[1].set_ylabel('Euler Angles (deg)')
    axes[1].legend(); axes[1].grid(True)

    axes[2].plot(t, actions[:, 0], label='T (N)')
    axes[2].set_ylabel('Thrust (N)')
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(t, actions[:, 1], label='tau_x (Nm)')
    axes[3].plot(t, actions[:, 2], label='tau_y (Nm)')
    axes[3].plot(t, actions[:, 3], label='tau_z (Nm)')
    axes[3].set_ylabel('Torques (Nm)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(); axes[3].grid(True)

    plt.tight_layout()
    plt.show()


def diagnose(scenario_name: str, seed: int = None):
    if seed is None:
        crashed = find_crashing_seeds(scenario_name)
        if not crashed:
            print(f"No crashes found in {scenario_name} for these 20 seeds.")
            return
        seed, ep_len = crashed[0]
    states, actions, terminated_early = _run_episode_with_actions(scenario_name, seed)
    print(f"seed={seed}, crashed at t={len(states)*0.01:.3f}s")
    _plot(states, actions, 0.01, f"PPO Diagnostic: {scenario_name}, seed={seed}")


diagnose("longrange_approach_wind")