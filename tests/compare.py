'''
Here we have our comparison tests between PID, LQR and PPO across two scenarios as defined below: 

SCENARIO 1 (local): 
Here we have the drone in a calm hover, initialised at +-0.3m from the target position. We use negligible wind of 0.0N as a baseline test, and then add the disturbance wind as defined in hover_env.py.
We do not include the embedded approach scenario here because close to the target position, we assume that the operator would be flying at negligible speed. 

SCENARIO 2 (longrange): 
Here we have the drone initialised further away from the target position (approximately 5m away from the target position). Again, we use negligible wind of 0.0N as a baseline test, and then add the disturbance wind as defined in hover_env.py.
As with the scenario above, we also include an embedded scenario here in which we have the quadrotor approaching the target position, initialised with an initial velocity, used to simulate the scenario where an operator is piloting the drone and wants the drone to reach hover position locally and stay there.

SCENARIO 3 (takeoff): 
Here we have the drone initialised at rest on the ground and we aim for the drone to reach the target position and maintain hover. We use negligible wind of 0.0N as a baseline test, and then add the disturbance wind as defined in hover_env.py.


----------

Metrics (we use per episode metrics, and then aggregate them over N_episodes)

- settle_rate: fraction of episodes that settle within a settle_threshold
- settle_mean: mean settling time (s) over settled episodes
- steadystate_error_mean: mean steady-state position error (m) over last number of steady_steps
- peak_error_mean: mean peak position error (m) over episode
- crash_rate: fraction of episodes that terminated (flipped/crashed into ground)
- effort_mean: mean per-step ||u-u_hover||^2 (which measures the control effort)

'''

from dataclasses import dataclass
import numpy as np
from typing import Optional

from envs.hover_env import HoverEnv
from controllers.pid import CascadedPIDController
from controllers.lqr import LQRController
from controllers.ppo import PPOController, MODEL_DIR
from quadrotor.params import Quadrotorparams

TARGET = np.array([0.0, 0.0, 1.0])
N_episodes = 20
seeds = list(range(N_episodes))
longrange_dist = 5.0 # (m) represnting the longrange scenario radius

settle_threshold = 0.10 # (m)
settle_window = 500 # number of steps (5s)
steady_steps = 500 # this represents the last 5s of that specific episode

@dataclass
class EpisodeResult:
    settling_time: Optional[float]
    steady_state_error: float
    peak_error: float
    control_effort: float
    crashed: bool
    episode_length: int
    positions: Optional[np.ndarray] = None
    error_trace: Optional[np.ndarray] = None

def _set_ic(env: HoverEnv, ic_type: str) -> np.ndarray:
    if ic_type == "longrange":
        azimuth = env.np_random.uniform(0,2*np.pi)
        elevation = env.np_random.uniform(-np.pi/4, np.pi/4)
        position_start = TARGET + longrange_dist*np.array([
            np.cos(elevation)*np.cos(azimuth),
            np.cos(elevation)*np.sin(azimuth),
            np.sin(elevation),
        ])
        position_start[2] = max(position_start[2], 0.3)
        env.state[0:3] = position_start
        env.state[3:12] = np.zeros(9)
    elif ic_type == "takeoff":
        env.state[:] = 0.0
        env.state[2] = 0.05
    return env.state.copy()

def run_episode(controller, env: HoverEnv, seed: int, ic_type: str, approach_speed: float = 0.0,
                record_trajectory: bool = False) -> EpisodeResult: 
    obs, _ = env.reset(seed=seed)
    if ic_type != "local":
        obs = _set_ic(env, ic_type)
    if approach_speed>0.0:
        azimuth = env.np_random.uniform(0,2*np.pi)
        elevation = env.np_random.uniform(-np.pi/4, np.pi/4)
        env.state[3] = approach_speed*np.cos(elevation)*np.cos(azimuth)
        env.state[4] = approach_speed*np.cos(elevation)*np.sin(azimuth)
        env.state[5] = approach_speed*np.sin(elevation)
        obs = env.state.copy()
    controller.reset()

    u_hover = np.array([env.params.m * env.params.g, 0.0, 0.0, 0.0])
    pos_errors = []
    efforts = []
    positions = [obs[0:3].copy()] if record_trajectory else None
    terminated = False

    for _ in range (env.max_steps):
        action = controller.compute_action(obs, TARGET, env.dt)
        obs, _, terminated, truncated, _ = env.step(action)
        # we find the position error and the control effort: 
        pos_errors.append(np.linalg.norm(obs[0:3]-TARGET))
        efforts.append(float(np.sum((action-u_hover)**2)))
        if record_trajectory:
            positions.append(obs[0:3].copy())
        if terminated or truncated:
            break
    
    pos_errors = np.array(pos_errors)
    n = len(pos_errors)

    settling_time = None
    for i in range(n-settle_window + 1):
        if np.all(pos_errors[i: i+settle_window] < settle_threshold):
            settling_time = i*env.dt
            break
    
    n_steady = min(steady_steps, n)
    return EpisodeResult(
        settling_time=settling_time,
        # we find the mean error over the last steady_steps or the whole episode if it is shorter; the maximum error during the episode; and the mean control_effort during the episode
        steady_state_error=float(np.mean(pos_errors[-n_steady:])),
        peak_error=float(np.max(pos_errors)),
        control_effort=float(np.mean(efforts)),
        crashed=bool(terminated),
        episode_length=n,
        positions=np.array(positions) if record_trajectory else None,
        error_trace=pos_errors if record_trajectory else None,
    )

# we run over all the seeds to produce different seeded episodeResults: 
def run_scenario(controller, env_kwargs: dict, ic_type: str, approach_speed: float=0.0) -> list[EpisodeResult]:
    return [
        run_episode(controller, HoverEnv(target=TARGET, **env_kwargs), seed, ic_type, approach_speed)
        for seed in seeds
    ]

# here we compute the summary stats for each controller on a particular scenario...we also want to filter out the episodes where the quadrotor never settled:
def aggregate(results: list[EpisodeResult]) -> dict: 
    settled = [r for r in results if r.settling_time is not None]
    return {
        "settle_rate": len(settled)/len(results),
        "settle_mean": float(np.mean([r.settling_time for r in settled])) if settled else float("nan"),
        "settle_std": float(np.std([r.settling_time for r in settled])) if settled else float("nan"),
        "ss_error_mean": float(np.mean([r.steady_state_error for r in results])),
        "peak_error_mean": float(np.mean([r.peak_error for r in results])),
        "crash_rate": float(np.mean([r.crashed for r in results])),
        "effort_mean": float(np.mean([r.control_effort for r in results])),
    }

def print_table(all_results: dict):
    col_w = 25
    metrics = [
        ("settle_rate", "Settle Percentage"),
        ("settle_mean", "Time to Settle (s)"),
        ("settle_std", "Settle Std Deviation (s)"),
        ("ss_error_mean", "Steady State Error (m)"),
        ("peak_error_mean", "Peak Error (m)"),
        ("crash_rate", "Crash Percentage"),
        ("effort_mean", "Control Effort"),
    ]
    for scenario, ctrl_results in all_results.items():
        print(f"\n{'='*80}\n Scenario: {scenario}\n{'='*80}")
        header = f"{'Controller':<14}" + "".join(f"{m[1]:>{col_w}}" for m in metrics)
        print(header)
        print("-"*len(header))
        for ctrl_name, results in ctrl_results.items():
            agg = aggregate(results)
            print(f"{ctrl_name:<14}" + "".join(f"{agg[m[0]]:>{col_w}.3f}" for m in metrics))

SCENARIOS = {
    "local_calm": {"env_kwargs": {"wind_magnitude": 0.0}, "ic": "local", "approach_speed": 0.0},
    "local_wind": {"env_kwargs": {"wind_magnitude": 2.0}, "ic": "local", "approach_speed": 0.0},
    "longrange_calm": {"env_kwargs": {"wind_magnitude": 0.0}, "ic": "longrange", "approach_speed": 0.0},
    "longrange_wind": {"env_kwargs": {"wind_magnitude": 2.0}, "ic": "longrange", "approach_speed": 0.0},
    "longrange_approach_calm": {"env_kwargs": {"wind_magnitude": 0.0}, "ic": "longrange", "approach_speed": 2.0},
    "longrange_approach_wind": {"env_kwargs": {"wind_magnitude": 2.0}, "ic": "longrange", "approach_speed": 2.0},
    "takeoff_calm": {"env_kwargs": {"wind_magnitude": 0.0}, "ic": "takeoff", "approach_speed": 0.0},
    "takeoff_wind": {"env_kwargs": {"wind_magnitude": 2.0}, "ic": "takeoff", "approach_speed": 0.0},
}

if __name__=="__main__":
    params = Quadrotorparams()
    PPO_PHASE5_DIR = "models/ppo_phase5"
    controllers = {
        "PID": CascadedPIDController(params),
        "LQR": LQRController(params),
        "PPO": PPOController(
            model_path=f"{PPO_PHASE5_DIR}/best_model",
            norm_path=f"{PPO_PHASE5_DIR}/best_vec_normalize.pkl"
        ),
    }

    all_results: dict = {}
    for scenario_name, configuration in SCENARIOS.items():
        all_results[scenario_name] = {}
        for controller_name, controller in controllers.items():
            print(f"Running {controller_name:>3s} | {scenario_name}...", end=" ", flush=True)
            results = run_scenario(controller, configuration["env_kwargs"], configuration["ic"], configuration["approach_speed"])
            all_results[scenario_name][controller_name] = results
            print(f"crash = {np.mean([r.crashed for r in results]):.0%}")
    
    print_table(all_results)

