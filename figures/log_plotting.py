import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
from envs.hover_env import HoverEnv
from controllers.pid import CascadedPIDController
from controllers.lqr import LQRController
from controllers.ppo import PPOController, MODEL_DIR
from quadrotor.params import Quadrotorparams
from compare import run_episode, aggregate, SCENARIOS, TARGET

# graph setup params: 
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.edgecolor": "#c3c2b7",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
  })

COLORS = {"PID": "#2a78d6", "LQR": "#008300", "PPO": "#e87ba4"}
LINESTYLES = {"PID": "-", "LQR": "--", "PPO": "-."}
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

def figure_3d_trajectory(controllers, scenario_name="local_calm", seed=0):
    config = SCENARIOS[scenario_name]
    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")

    for name, controller in controllers.items():
        env = HoverEnv(target=TARGET, **config["env_kwargs"])
        result = run_episode(controller, env, seed, config["ic"], config["approach_speed"],
                            record_trajectory=True)
        pos = result.positions
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                color=COLORS[name], linestyle=LINESTYLES[name], linewidth=1.5, label=name)
        ax.scatter(*pos[0], color=COLORS[name], marker="o", s=25)

    ax.scatter(*TARGET, color="#0b0b0b", marker="*", s=120, label="Target", zorder=5)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(f"3D Trajectory - {scenario_name}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/trajectory_3d_{scenario_name}.pdf")
    plt.close(fig)

def figure_distance_density(controllers, scenario_name="local_calm", n_episodes=20):
    config = SCENARIOS[scenario_name]
    fig, ax = plt.subplots(figsize=(6, 4))

    for name, controller in controllers.items():
        all_errors = []
        for seed in range(n_episodes):
            env = HoverEnv(target=TARGET, **config["env_kwargs"])
            result = run_episode(controller, env, seed, config["ic"], config["approach_speed"],
                                record_trajectory=True)
            all_errors.append(result.error_trace[-500:]) 
        all_errors = np.concatenate(all_errors)

        if all_errors.std() < 1e-8:  
            ax.axvline(all_errors.mean(), color=COLORS[name], linestyle=LINESTYLES[name],
                        linewidth=1.8, label=f"{name} (constant \u2248 {all_errors.mean():.4f} m)")
        else:
            kde = gaussian_kde(all_errors)
            xs = np.linspace(0, max(all_errors.max(), 0.05), 300)
            ax.plot(xs, kde(xs), color=COLORS[name], linestyle=LINESTYLES[name], linewidth=1.8, label=name)
            ax.fill_between(xs, kde(xs), color=COLORS[name], alpha=0.12)

    ax.set_xlabel("Distance from target (m)")
    ax.set_ylabel("Density")
    ax.set_title(f"Steady-State Tracking Error Distribution - {scenario_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/distance_density_{scenario_name}.pdf")
    plt.close(fig)

def figure_error_vs_time(controllers, scenario_name="local_calm", n_episodes=20, horizon_s=10.0):
    config = SCENARIOS[scenario_name]
    fig, ax = plt.subplots(figsize=(7, 4))

    env_probe = HoverEnv(target=TARGET, **config["env_kwargs"])
    horizon_steps = int(horizon_s / env_probe.dt)

    for name, controller in controllers.items():
        traces = np.full((n_episodes, horizon_steps), np.nan)
        for seed in range(n_episodes):
            env = HoverEnv(target=TARGET, **config["env_kwargs"])
            result = run_episode(controller, env, seed, config["ic"], config["approach_speed"],
                                record_trajectory=True)
            n = min(len(result.error_trace), horizon_steps)
            traces[seed, :n] = result.error_trace[:n]

        t_axis = np.arange(horizon_steps) * env_probe.dt
        mean_err = np.nanmean(traces, axis=0)
        std_err = np.nanstd(traces, axis=0)

        ax.plot(t_axis, mean_err, color=COLORS[name], linestyle=LINESTYLES[name], linewidth=1.5, label=name)
        ax.fill_between(t_axis, mean_err - std_err, mean_err + std_err, color=COLORS[name], alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Position error (m)")
    ax.set_title(f"Tracking Error Over Time (mean \u00b1 std, n={n_episodes}) - {scenario_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/error_vs_time_{scenario_name}.pdf")
    plt.close(fig)

def figure_summary_bars(all_results, metric_key="settle_rate", ylabel="Settle Rate", fname="settle_rate_summary"):
    scenario_names = list(all_results.keys())
    controller_names = list(next(iter(all_results.values())).keys())

    x = np.arange(len(scenario_names))
    width = 0.8 / len(controller_names)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    for i, name in enumerate(controller_names):
        values = [aggregate(all_results[s][name])[metric_key] for s in scenario_names]
        ax.bar(x + i * width, values, width, color=COLORS[name], label=name)

    ax.set_xticks(x + width * (len(controller_names) - 1) / 2)
    ax.set_xticklabels(scenario_names, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by Scenario")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{FIG_DIR}/{fname}.pdf")
    plt.close(fig)

if __name__ == "__main__":
    params = Quadrotorparams()
    PPO_PHASE5_DIR = "models/ppo_phase5"
    controllers = {
        "PID": CascadedPIDController(params),
        "LQR": LQRController(params),
        "PPO": PPOController(model_path=f"{PPO_PHASE5_DIR}/best_model", norm_path=f"{PPO_PHASE5_DIR}/best_vec_normalize.pkl"),
    }

    figure_3d_trajectory(controllers, scenario_name="takeoff_calm")
    figure_distance_density(controllers, scenario_name="takeoff_calm")
    figure_error_vs_time(controllers, scenario_name="takeoff_calm")


#for phase in ["ppo_phase3", "ppo_phase3", "ppo_phase5"]:
#    d = np.load(f"logs/{phase}/evaluations.npz")
#    timesteps = d["timesteps"]
#    mean_reward = d["results"].mean(axis=1)
#    mean_ep_len = d["ep_lengths"].mean(axis=1)
#
#    best = np.argmax(mean_reward)
#    longest = np.argmax(mean_ep_len)

#    print(f"=== {phase} ===")
#    print(f"EvalCallback's 'best' (highest reward): step {timesteps[best]}, "
#          f"reward={mean_reward[best]:.1f}, ep_len={mean_ep_len[best]:.1f}")
#    print(f"Actual longest-surviving checkpoint: step {timesteps[longest]}, "
#          f"reward={mean_reward[longest]:.1f}, ep_len={mean_ep_len[longest]:.1f}")
#    print()

#   fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
#   fig.suptitle(phase)
#    ax1.plot(timesteps, mean_reward)
#    ax1.axvline(timesteps[best], color="g", linestyle="--", label="'best' policy")
#    ax1.set_ylabel("eval mean reward")
#    ax1.legend()
#    ax2.plot(timesteps, mean_ep_len)
#    ax2.axhline(5000, color="r", linestyle=":", label="full episode")
#    ax2.axvline(timesteps[best], color="g", linestyle="--")
#    ax2.set_ylabel("eval mean ep_len")
#    ax2.set_xlabel("timestep")
#    ax2.legend()
#   plt.show()
