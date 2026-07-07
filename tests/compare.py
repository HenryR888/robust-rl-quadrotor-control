'''
Here we have our comparison tests between PID, LQR and PPO across two scenarios as defined below: 

SCENARIO 1 (local): 
Here we have the drone in a calm hover, initialised at +-0.3m from the target position. We use negligible wind of 0.0N as a baseline test, and then add the disturbance wind as defined in hover_env.py.
We also include an embedded scenario here in which we have the quadrotor approaching the target position, initialised with an initial velocity, used to simulate the scenario where an operator is piloting the drone and wants the drone to reach hover position locally and stay there.

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
settle_window = 50 # number of steps (0.5s)
steady_steps = 500 # this represents the last 5s of that specific episode

@dataclass
class EpisodeResult:
    settling_time: Optional[float]
    steady_state_error: float
    peak_error: float
    control_effort: float
    crashed: bool
    episode_length: int

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