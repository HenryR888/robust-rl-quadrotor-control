'''
Here we write some validation tests for the tuned Cascaded PID Controller within the HoverEnv 
Each test asserts a quantitative pass/fail criteria and we produce a traj. plot to in order to validate our assertions.
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv
from controllers.pid import CascadedPIDController

target = np.array([0.0,0.0,1.0]) # our target here is 1m
settle_time = 3.0 # in seconds...allow controller to settle before checking steady-state error
position_tol = 0.05 # in metres...acceptable steady-state position error

def _run_episode(initial_state: np.ndarray, sim_time: float = 10.0) -> tuple[np.ndarray, np.ndarray, bool]:
    '''
    run simulation for 10s
    '''
    # initialise environment from HoverEnv:
    env = HoverEnv(target=target)
    controller = CascadedPIDController(env.params)

    env.reset()
    env.state = initial_state.copy()
    obs = initial_state.copy()
    obs = env.state.copy()
    controller.reset()

    n_steps = int(sim_time / env.dt)
    states, actions = [], []
    terminated_early = False

    for _ in range(n_steps):
        action = controller.compute_action(obs, env.target, env.dt)
        obs,_,terminated,_,_ = env.step(action)
        states.append(obs.copy())
        actions.append(action.copy())
        if terminated:
            terminated_early = True
            break
    return np.array(states), np.array(actions), terminated_early

def _settle_index(env_dt: float) -> int:
    return int(settle_time/env_dt)


# plots: 
def _plot(states: np.ndarray, actions: np.ndarray, dt: float, title: str):
    t = np.arange(len(states))*dt
    fig, axes = plt.subplots(4,1,figsize=(10,12))
    fig.suptitle(title)

    # plot position: 
    axes[0].plot(t, states[:, 0], label='x')
    axes[0].plot(t, states[:, 1], label='y')
    axes[0].plot(t, states[:, 2], label='z')
    axes[0].axhline(target[2], color='k', linestyle='--', linewidth=0.8, label='z target')
    axes[0].set_ylabel('Position (m)')
    axes[0].legend(); axes[0].grid(True)

    # euler angle plot: 
    axes[1].plot(t, np.degrees(states[:, 6]), label='phi (roll)')
    axes[1].plot(t, np.degrees(states[:, 7]), label='theta (pitch)')
    axes[1].plot(t, np.degrees(states[:, 8]), label='psi (yaw)')
    axes[1].set_ylabel('Euler Angles (deg)')
    axes[1].legend(); axes[1].grid(True)

    # thrust plot: 
    axes[2].plot(t, actions[:, 0], label='T (N)')
    axes[2].set_ylabel('Thrust (N)')
    axes[2].set_xlabel('Time (s)')  
    axes[2].legend(); axes[2].grid(True)

    axes[3].plot(t, actions[:, 1], label='tau_x (Nm)')
    axes[3].plot(t, actions[:, 2], label='tau_y (Nm)')
    axes[3].plot(t, actions[:, 3], label='tau_z (Nm)')
    axes[3].set_ylabel('Torques (Nm)')
    axes[3].set_xlabel('Time (s)')
    axes[3].legend(); axes[3].grid(True)

    plt.tight_layout() 
    plt.show()


def test_hover_stability():
    '''
    we start the drone at target position, and it should fly for sim_time without crash/flip and maintain position.
    '''
    initial = np.array([0.0,0.0,1.0,
                        0.0,0.0,0.0,
                        0.0,0.0,0.0,
                        0.0,0.0,0.0])
    states, actions, terminated_early = _run_episode(initial)

    _plot(states, actions, 0.01, 'Test 1: Hover Stability')

    assert not terminated_early, "the drone crashed or flipped during hover stability test"
    final_pos_error = np.linalg.norm(states[-1,0:3] - target)
    assert final_pos_error<position_tol, f"Steady-state position error too large: {final_pos_error:.4f} m"

    
def test_z_step_recovery():
    '''
    Drone starts 0.5m below target. Should climb and hold position at z=1.0 within settle_time.
    '''
    initial = np.array([0.0,0.0,0.5,
                        0.0,0.0,0.0,
                        0.0,0.0,0.0,
                        0.0,0.0,0.0])
    states, actions, terminated_early = _run_episode(initial)

    _plot(states, actions, 0.01, 'Test 2: Z Recovery (start 0.5m below target)')
    assert not terminated_early, "Drone crashed or flipped during z recovery"
    settle_idx = _settle_index(0.01)
    after_settle_z_errors = np.abs(states[settle_idx:, 2] - target[2])
    assert after_settle_z_errors.max()<position_tol, (
        f"z did not settle within {position_tol}m after {settle_time}s — max error: {after_settle_z_errors.max():.4f} m"
    )

def test_attitude_recovery():
    '''
    Here we start drone with pi/18 rad (10 deg) roll perturbation and we expect drone to damp without flipping
    '''
    initial = np.array([0.0,0.0,1.0,
                        0.0,0.0,0.0,
                        np.pi/18,0.0,0.0,
                        0.0,0.0,0.0])
    states, actions, terminated_early = _run_episode(initial)

    _plot(states, actions, 0.01, 'Test 4: Attitude Recovery (pi/18 roll perturbation)')

    assert not terminated_early, "Drone crashed or flipped during attitude recovery"
    settle_idx = _settle_index(0.01)
    after_settle_roll = np.abs(states[settle_idx:, 6])
    assert after_settle_roll.max() < np.pi/36, (
        f"Roll did not damp below pi/36 rads after {settle_time}s — max: {np.degrees(after_settle_roll.max()):.2f} rads"
    )

# 3 different tests: 

#test_hover_stability()
#test_z_step_recovery()
test_attitude_recovery()

        