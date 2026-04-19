'''
Here we implement an initial test to see that the cascaded PID controller implementation works within the HoverEnv. 
We utilise this file to plot trajectories for our control outputs and tune our gains Kp, Ki and Kd respectively for our PID controller. 
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv
from controllers.pid import CascadedPIDController

env = HoverEnv(target=np.array([0.0,0.0,1.0])) # target here is 1.0m
controller = CascadedPIDController(env.params)

# initialise environment from HoverEnv: 
obs, _ = env.reset()
env.state = np.array( # start the drone at  hover positions 1.0m 
    [0.0,0.0,1.0,
     0.0, 0.0,0.0,
     0.0,0.1,0.0,
     0.0,0.0,0.0,])
obs = env.state.copy()
controller.reset()

states = []
actions = []

for _ in range(env.max_steps):
    action = controller.compute_action(obs, env.target, env.dt)
    obs, _, terminated, truncated, _ = env.step(action)
    states.append(obs)
    actions.append(action.copy())
    # comment out for tuning: 
    #if terminated or truncated:
       #break

states = np.array(states)
actions = np.array(actions)
t = np.arange(len(states)) * env.dt


# make the plots: 

fig, axes = plt.subplots(3,1, figsize=(10,9))

# plot position: 
axes[0].plot(t,states[:,0], label = 'x')
axes[0].plot(t, states[:, 1], label='y')
axes[0].plot(t, states[:, 2], label='z')
axes[0].axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='z target')
axes[0].set_ylabel('Position (m)')
axes[0].legend(); axes[0].grid(True)


# Euler angles (attitude): 
axes[1].plot(t, states[:, 6], label='phi (roll)')
axes[1].plot(t, states[:, 7], label='theta (pitch)')
axes[1].plot(t, states[:, 8], label='psi (yaw)')
axes[1].set_ylabel('Euler angles (attitude)')
axes[1].legend(); axes[1].grid(True)

# control variable outputs: 
axes[2].plot(t, actions[:, 0], label='T (N)')
axes[2].plot(t, actions[:, 1], label='tau_x (Nm)')
axes[2].plot(t, actions[:, 2], label='tau_y (Nm)')
axes[2].plot(t, actions[:, 3], label='tau_z (Nm)')
axes[2].axhline(env.params.m * env.params.g, color='k', linestyle='--', linewidth=0.8, label='T_hover')
axes[2].set_ylabel('Control Variable Outputs')
axes[2].set_xlabel('Time (s)')
axes[2].legend(); axes[2].grid(True)

plt.tight_layout()
plt.show()
