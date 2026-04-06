'''
Here we apply a periodic thrust test using HoverEnv.step()

We apply T = mg + A.sin(nu.t) - i..e we are hovering and adding some sinuosoidal variation to our thrust input. 

We expect that: a_z(t) = (A/m).sin(nu.t) - i.e. sinusoidal change in acceleration.
- v_z(t) = (A/(m.nu))(1-cos(nut)) after integration (Recall that ICs are zdot(0) = 0, and z(0) = z_0)
- Integrating again, we have that: z = z_0 + (A/m.nu)t - (a/m.nu^2)sin(nu.t)

Again we maintain phi, theta, psi, omega at zero 
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv

# initiliase the env. with some constants:
env = HoverEnv()
params = env.params
mg = params.m * params.g
z_0 = 5.0 # initialise the drone at 5m, same as hover test. 
A = 0.3*mg # thrust amp...we can vary this as desired. 
nu = 2*np.pi # this corresponds to oscillation frequency, and thus we have 1 cycle per second

env.reset()
env.state = np.array([
        0.0, 0.0, z_0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
])

states = [env.state.copy()]
thrusts = [mg + A*np.sin(0.0)] # initial thrust list and states list

for step in range(1, env.max_steps + 1):
    t_current = step*env.dt # actual clock time in env. 
    T = mg + A * np.sin(nu*t_current)
    action = np.array([T, 0.0, 0.0, 0.0])
    obs, _, terminated, truncated, _ = env.step(action)
    states.append(obs)
    thrusts.append(T)
    if terminated or truncated:
        break

states = np.array(states)
thrusts = np.array(thrusts)
t = np.arange(len(states))*env.dt # convert the discrete time indices to real time indices...recall that dt = 0.01s

z = states[:, 2]
vz = states[:, 5]
az = np.gradient(vz, env.dt) # here we numerically differentiate v_z(t) to obtain the acceleration

# now we find the analytical solutions for accel. vz and z: 
az_analytical = (A/params.m) * np.sin(nu*t)
vz_analytical = (A/(params.m*nu)) * (1 - np.cos(nu*t))
z_analytical = z_0 + (A/(params.m*nu))*t - (A/(params.m* nu**2)) * np.sin(nu*t)

fig, axes = plt.subplots(2, 2, figsize=(11, 7)) 
fig.suptitle(
    r'Periodic Thrust Test: $T = mg + A\sin(\nu t)$, '
    r'$A = 0.3\,mg$, $\nu = 2\pi\,\mathrm{rad/s}$',
    fontsize=12, fontweight='bold'
)

EXPECTED = {'color': 'tomato', 'linestyle': '--', 'linewidth': 1.0}

def style_ax(ax):                                                                                                                                                                                                   
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)                                                                                                                                                                                     
    ax.legend(fontsize=8, loc='best')
    ax.grid(True, linestyle=':', alpha=0.6)

ax = axes[0, 0]
ax.plot(t, thrusts, color='steelblue', linewidth=1.5, label=r'$T(t)$')                                                                                                                                              
ax.axhline(mg, **EXPECTED, label=r'$mg$ (hover)')                                                                                                                                                                 
ax.set_ylabel('Thrust (N)', fontsize=9)                                                                                                                                                                             
ax.set_title('Applied Thrust', fontsize=10)                                                                                                                                                                         
style_ax(ax)

ax = axes[0, 1] 
ax.plot(t, az, color='steelblue', linewidth=1.5, label=r'$a_z(t)$ simulated')
ax.plot(t, az_analytical, **EXPECTED, label=r'$(A/m)\sin(\nu t)$')                                                                                                                                                
ax.set_ylabel(r'$a_z$ (m/s$^{2}$)', fontsize=9)                                                                                                                                                                    
ax.set_title('Vertical Acceleration', fontsize=10)                                                                                                                                                                  
style_ax(ax)

ax = axes[1, 0]                                                                                                                                                                                                     
ax.plot(t, vz, color='steelblue', linewidth=1.5, label=r'$v_z(t)$ simulated')
ax.plot(t, vz_analytical, **EXPECTED, label=r'$(A/m\nu)(1-\cos(\nu t))$')                                                                                                                                         
ax.set_ylabel(r'$v_z$ (m/s)', fontsize=9)                                                                                                                                                                    
ax.set_title('Vertical Velocity', fontsize=10)                                                                                                                                                                      
style_ax(ax)

ax = axes[1, 1] 
ax.plot(t, z, color='steelblue', linewidth=1.5, label=r'$z(t)$ simulated')
ax.plot(t, z_analytical, **EXPECTED, label=r'expected')                                                                                                                                                         
ax.set_ylabel('$z$ (m)', fontsize=9)
ax.set_title('Elevation', fontsize=10)                                                                                                                                                                               
style_ax(ax)

plt.tight_layout()
plt.savefig('figures/periodic_thrust_test.png', dpi=150, bbox_inches='tight')                                                                                                                                       
plt.show()
