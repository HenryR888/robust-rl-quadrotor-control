'''
Here we implement an initial test to see that the cascaded PID controller implementation works within the HoverEnv. 
We utilise this file to plot trajectories for our control outputs and tune our gains Kp, Ki and Kd respectively for our PID controller. 
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv
from controllers.pid import CascadedPIDController

env_calm = HoverEnv(target=np.array([0.0,0.0,1.0])) # target here is 1.0m
env_wind = HoverEnv(target=np.array([0.0,0.0,1.0]), wind_magnitude=2.0)
controller = CascadedPIDController(env_calm.params)

# initialise environment from HoverEnv: 
obs, _ = env_calm.reset(seed=0)
env_calm.state = np.array(
    [0.2,0.2,0.8,
     0.0, 0.0,0.0,
     0.0,0.0,0.0,
     0.0,0.0,0.0,])
obs = env_calm.state.copy()
controller.reset()

states_calm = []
actions_calm = []

for _ in range(env_calm.max_steps):
    action = controller.compute_action(obs, env_calm.target, env_calm.dt)
    obs, _, terminated, truncated, _ = env_calm.step(action)
    states_calm.append(obs)
    actions_calm.append(action.copy())
    #comment out for tuning: 
    #if terminated or truncated:
        #break

states_calm = np.array(states_calm)
actions_calm = np.array(actions_calm)
t_calm = np.arange(len(states_calm)) * env_calm.dt



obs, _ = env_wind.reset(seed=0)
env_wind.state = np.array([0.2, 0.2, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
obs = env_wind.state.copy()
controller.reset()

states_wind = []
actions_wind = []
for _ in range(env_wind.max_steps):
    action = controller.compute_action(obs, env_wind.target, env_wind.dt)
    obs, _, terminated, truncated, _ = env_wind.step(action)
    states_wind.append(obs)
    actions_wind.append(action.copy())
    # comment out for tuning: 
    #if terminated or truncated:
        #break

states_wind = np.array(states_wind)
actions_wind = np.array(actions_wind)
t_wind = np.arange(len(states_wind))*env_wind.dt


# make the plots: 

fig, axes = plt.subplots(3,2, figsize=(14,9))

# calm column:
# (Position):  
axes[0][0].plot(t_calm, states_calm[:,0], label='x')
axes[0][0].plot(t_calm, states_calm[:,1], label='y')
axes[0][0].plot(t_calm, states_calm[:,2], label='z')
axes[0][0].axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='z target')
axes[0][0].set_title('Calm')
axes[0][0].set_ylabel('Position (m)')
axes[0][0].legend(); axes[0][0].grid(True)

# (Euler angles): 
axes[1][0].plot(t_calm, states_calm[:,6], label='phi (roll)')
axes[1][0].plot(t_calm, states_calm[:,7], label='theta (pitch)')
axes[1][0].plot(t_calm, states_calm[:,8], label='psi (yaw)')
axes[1][0].set_ylabel('Euler angles (rad)')
axes[1][0].legend(); axes[1][0].grid(True)

# (Control Variable Outputs): 
axes[2][0].plot(t_calm, actions_calm[:,0], label='T (N)')
axes[2][0].plot(t_calm, actions_calm[:,1], label='tau_x (Nm)')
axes[2][0].plot(t_calm, actions_calm[:,2], label='tau_y (Nm)')
axes[2][0].plot(t_calm, actions_calm[:,3], label='tau_z (Nm)')
axes[2][0].axhline(env_calm.params.m * env_calm.params.g, color='k', linestyle='--', linewidth=0.8, label='T_hover')
axes[2][0].set_ylabel('Control Variable Outputs')
axes[2][0].set_xlabel('Time (s)')
axes[2][0].legend(); axes[2][0].grid(True)

# Wind column: 
# (Position): 
axes[0][1].plot(t_wind, states_wind[:,0], label='x')
axes[0][1].plot(t_wind, states_wind[:,1], label='y')
axes[0][1].plot(t_wind, states_wind[:,2], label='z')
axes[0][1].axhline(1.0, color='k', linestyle='--', linewidth=0.8, label='z target')
axes[0][1].set_title('Wind (2.0 N)')
axes[0][1].legend(); axes[0][1].grid(True)

# (Euler angles): 
axes[1][1].plot(t_wind, states_wind[:,6], label='phi (roll)')
axes[1][1].plot(t_wind, states_wind[:,7], label='theta (pitch)')
axes[1][1].plot(t_wind, states_wind[:,8], label='psi (yaw)')
axes[1][1].legend(); axes[1][1].grid(True)

# (Control Variable Outputs):
axes[2][1].plot(t_wind, actions_wind[:,0], label='T (N)')
axes[2][1].plot(t_wind, actions_wind[:,1], label='tau_x (Nm)')
axes[2][1].plot(t_wind, actions_wind[:,2], label='tau_y (Nm)')
axes[2][1].plot(t_wind, actions_wind[:,3], label='tau_z (Nm)')
axes[2][1].axhline(env_wind.params.m * env_wind.params.g, color='k', linestyle='--', linewidth=0.8, label='T_hover')
axes[2][1].set_xlabel('Time (s)')
axes[2][1].legend(); axes[2][1].grid(True)

plt.tight_layout()
plt.show()
