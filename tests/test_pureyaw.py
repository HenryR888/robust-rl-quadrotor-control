'''
Here we execute a pure yaw test using HoverEnv.step()

We apply T=mg and a constant roll torque: tau_x = tau_0, with tau_y = tau_z=0 (that is no yaw torque nor pitch torque)

Our expected behaviour is as follows: 
- omega_dot_x(t) = tau_0/I_x (constant angular acceleration)
- omega_x(t) = (tau_0/I_x)t (linear angular velocity)
- phi(t) = (tau_0/2.I_x).t^2 (quad. profile of roll angle)
- theta, psi, omega_y, omega_z all remain approx 0. 
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv

# initiliase the env. with some constants:
env = HoverEnv()
params = env.params
mg = params.m * params.g
z_0 = 5.0 # initialise the drone at 5m, same as hover test.

tau_0 = 0.05 * params.tau_z_max # 5% of max roll torque

env.reset()
env.state = np.array([
        0.0, 0.0, z_0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
])

action = np.array([mg, 0.0, 0.0, tau_0])
states = [env.state.copy()]

for _ in range(env.max_steps):
    obs, _, terminated, truncated, _ = env.step(action)
    states.append(obs)
    if terminated or truncated:
        break

states = np.array(states)
t = np.arange(len(states))*env.dt # convert the discrete time indices to real time indices...recall that dt = 0.01s

phi = states[:, 8]
omega_x = states[:, 11]
omega_dot_x = np.gradient(omega_x, env.dt)

# now we find the analytical solutions for ang. accel. ang. velocity and phi: 

alpha_analytical = np.full_like(t, tau_0/params.I_z)
omega_analytical = (tau_0/params.I_z)*t
phi_analytical = (tau_0/(2*params.I_z))*(t**2)

# plot the figs: 
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle(
    r'Pure Roll Test: constant $\tau_x = \tau_0$, $T = mg$',
    fontsize=12, fontweight='bold'
)

EXPECTED = {'color': 'tomato', 'linestyle': '--', 'linewidth': 1.0}                                                                                                                                               
   
def style_ax(ax):                                                                                                                                                                                                   
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

# ang. acceleration graph: 
ax = axes[0]                                                                                                                                                                                                        
ax.plot(t, omega_dot_x, color='steelblue', linewidth=1.5, label=r'$\dot{\omega}_x$ simulated')
ax.plot(t, alpha_analytical, **EXPECTED, label=rf'$(\tau_0/I_x) = {tau_0/params.I_z:.3f}$ rad/s$^{{2}}$')                                                                                                                    
ax.set_ylabel(r'$\dot{\omega}_x$ (rad/s$^{2}$)', fontsize=9)                                                                                                                                                             
ax.set_title('Roll Angular Acceleration', fontsize=10)                                                                                                                                                              
style_ax(ax)
axes[0].set_ylim(tau_0/params.I_z - 0.01, tau_0/params.I_z + 0.01)

# ang. velocity graph: 
ax = axes[1]    
ax.plot(t, omega_x, color='steelblue', linewidth=1.5, label=r'$\omega_x(t)$ simulated')
ax.plot(t, omega_analytical, **EXPECTED, label=r'$(\tau_0/I_x)\,t$')                                                                                                                                                   
ax.set_ylabel(r'$\omega_x$ (rad/s)', fontsize=9)                                                                                                                                                             
ax.set_title('Roll Angular Velocity', fontsize=10)                                                                                                                                                                  
style_ax(ax)

# roll angle graph: 
ax = axes[2]                                                                                                                                                                                                        
ax.plot(t, phi, color='steelblue', linewidth=1.5, label=r'$\phi(t)$ simulated')
ax.plot(t, phi_analytical, **EXPECTED, label=r'$(\tau_0/2I_x)\,t^2$')                                                                                                                                                  
ax.set_ylabel(r'$\phi$ (rad)', fontsize=9)
ax.set_title('Roll Angle', fontsize=10)                                                                                                                                                                             
style_ax(ax)

plt.tight_layout()
plt.savefig('figures/pure_yaw_test.png', dpi=150, bbox_inches='tight')
plt.show()