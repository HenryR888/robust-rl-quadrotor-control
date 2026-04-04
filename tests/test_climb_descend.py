'''
Here we complete a climb and descent test using the HoverEnv.step() function. 
We note that to climb, T>mg, which should result in constant up accel. 
And to descend, T<mg, which should result in a constant downward accel. 

We expect to have acceleration in z be constant (a_z(t) = (T-mg)/m)
- v_z(t) = a_z.t (linear in time)
- z(t) = z_0 + 0.5.a_z.t^2 (quadratic in time)
- phi, theta, psi, omega all remain 0. 
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv

# initiliase the env. with some constants:
env = HoverEnv()
params = env.params
mg = params.m * params.g
z_0 = 5.0 # initialise the drone at 5m, same as hover test. 

T_climb = 1.5 * 4.0 * params.hover_thrust # set the climb thrust to 1.5 x hover thrust
T_descent = 0.5 * 4.0 * params.hover_thrust # set descent thrust to 0.5 x hover thrust

def run_step(T_total, z0=z_0):
    env = HoverEnv()
    env.reset()
    env.state = np.array([
        0.0, 0.0, z0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])
    action = np.array([T_total, 0.0, 0.0, 0.0])
    states = [env.state.copy()]
    for _ in range(env.max_steps):
        obs, _, terminated, truncated, _ = env.step(action)
        states.append(obs)
        if terminated or truncated:
            break
    states = np.array(states)
    t = np.arange(len(states)) * env.dt # convert the discrete time indices to real time indices...recall that dt = 0.01s
    z = states[:,2]
    vz = states[:,5]
    az = np.gradient(vz, env.dt) # here we numerically differentiate v_z(t) to obtain the acceleration
    return t, z, vz, az
    
# get the respective values from the run_step function, when applying climbing thrust and descent thrust: 
t_climb, z_climb, vz_climb, az_climb = run_step(T_climb)
t_descent, z_descent, vz_descent, az_descent = run_step(T_descent)

# compute the constants for climb and descent: 
a_climb = (T_climb-mg)/params.m
a_descent = (T_descent-mg)/params.m

# plot the figs: 
fig, axes = plt.subplots(3, 2, figsize=(11, 9))
fig.suptitle('Climb and Descent Test', fontsize=13, fontweight='bold')

EXPECTED = {'color': 'tomato', 'linestyle': '--', 'linewidth': 1.0}

def style_ax(ax):                                                                                                                                                                        
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=8, loc='best')                                                                                                                                                    
    ax.grid(True, linestyle=':', alpha=0.6)

# set the title for the left and right columns of the graphs: 
axes[0, 0].set_title(r'Climb ($T = 1.5\,mg$)', fontsize=10)                                                                                                                              
axes[0, 1].set_title(r'Descent ($T = 0.5\,mg$)', fontsize=10)


# elevation graphs: 
for ax, t, z, a in [(axes[0, 0], t_climb, z_climb, a_climb),                                                                                                                                     
                    (axes[0, 1], t_descent, z_descent, a_descent)]:                                                                                                                                  
    ax.plot(t, z, color='steelblue', linewidth=1.5, label=r'$z(t)$ simulated')
    ax.plot(t, z_0 + 0.5*a*t**2, **EXPECTED, label=r'$z_0 + \frac{1}{2}a_z t^2$')                                                                                                        
    ax.set_ylabel('Elevation $z$ (m)', fontsize=9)                                                                                                                                                 
    style_ax(ax)

# vertical linear velocity graphs: 
for ax, t, vz, a in [(axes[1, 0], t_climb, vz_climb, a_climb),
                    (axes[1, 1], t_descent, vz_descent, a_descent)]:                                                                                                                                
    ax.plot(t, vz, color='steelblue', linewidth=1.5, label=r'$v_z(t)$ simulated')                                                                                                        
    ax.plot(t, a*t, **EXPECTED, label=r'$a_z \cdot t$')
    ax.set_ylabel(r'Vertical Linear Velocity $v_z$ (m/s)', fontsize=9)                                                                                                                                     
    style_ax(ax)

# acceleration graphs: 
for ax, t, az, a in [(axes[2, 0], t_climb, az_climb, a_climb),                                                                                                                                   
                    (axes[2, 1], t_descent, az_descent, a_descent)]:
    ax.plot(t, az, color='steelblue', linewidth=1.5, label=r'$a_z(t)$ simulated')                                                                                                        
    ax.axhline(a, **EXPECTED, label=f'Expected = {a:.3f} m s$^{{-2}}$')
    ax.set_ylabel(r'Vertical Acceleration $a_z$ (m/s$^{2}$)', fontsize=9)                                                                                                                                     
    style_ax(ax) 

plt.tight_layout()
plt.savefig('figures/climb_descent_test.png', dpi=150, bbox_inches='tight')
plt.show()