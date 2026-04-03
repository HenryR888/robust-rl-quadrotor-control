'''
Here we perform a hover test using the HoverEnv.step(). 
We Initialise the drone at [0,0,5] and apply constant hover thrust: [T=mg,0,0,0] for every step and verify that: 
z(t)=5; v_z(t)=0 for all t, phi=theta=psi=0 and angular velocity =0
'''

import numpy as np
import matplotlib.pyplot as plt
from envs.hover_env import HoverEnv

# initialise environment from HoverEnv: 
env = HoverEnv(target=np.array([0.0,0.0,5.0]))
env.reset()

env.state = np.array([
    0.0,0.0, 5.0, # start the drone hover at 5.0m mark
    0.0, 0.0, 0.0, 
    0.0, 0.0, 0.0,
    0.0, 0.0, 0.0,
])

params = env.params
hover_action = np.array([params.hover_thrust*4, 0.0, 0.0, 0.0]) # [T, tau_x, tau_y, tau_z]


states = [env.state.copy()] # here we start from the IC of the iniital state that we initialise in                                                                                                                                                                                        
   
for _ in range(env.max_steps): # step through the hover_env                                                                                                                                                                                   
    obs, _, terminated, truncated, _ = env.step(hover_action)
    states.append(obs.copy())                                                                                                                                                                                       
    if terminated or truncated:
        break                                                                                                                                                                                                       
                  
states = np.array(states)
t = np.arange(len(states)) * env.dt # convert the discrete time indices to real time indices...recall that dt = 0.01s
                                                                                                                                                                                                                      
z     = states[:, 2]
vz    = states[:, 5]                                                                                                                                                                                                
phi   = states[:, 6]                                                                                                                                                                                                
theta = states[:, 7]
psi   = states[:, 8]                                                                                                                                                                                                
omega_x   = states[:, 9]
omega_y   = states[:, 10]                                                                                                                                                                                               
omemga_z   = states[:, 11]

# we now check to see numerically if any of the tests have failed: 

tol = 1e-6                                                                                                                                                                                                          
assert np.allclose(z, 5.0, atol=tol), f"elevation drifted: {np.max(np.abs(z - 5.0)):.2e} m"
assert np.allclose(vz,0.0, atol=tol), f"v_z drifted: {np.max(np.abs(vz)):.2e} m/s"                                                                                                                              
assert np.allclose(phi,0.0, atol=tol), "phi drifted"
assert np.allclose(theta,0.0, atol=tol), "theta drifted"                                                                                                                                                           
assert np.allclose(psi, 0.0, atol=tol), "psi drifted"                                                                                                                                                             
assert np.allclose(omega_x, 0.0, atol=tol), "omega_x drifted"                                                                                                                                                         
assert np.allclose(omega_y,0.0, atol=tol), "omega_y drifted"                                                                                                                                                         
assert np.allclose(omemga_z,0.0, atol=tol), "omega_z drifted"
                                                                                                                                                                                                                      
print("All hover checks passed.")
                                                                                                                                                                                                                      
# Make the plots:                                                                                                                                    
   
fig, axes = plt.subplots(2, 2, figsize=(11, 7))                                                                                                                                                                     
fig.suptitle(   
      r"Hover Test: $z_0 = 5\,\mathrm{m}$, constant thrust $T = mg$",                                                                                                                                                 
      fontsize=13, fontweight='bold'                                                                                                                                                                                  
)                                                                                                                                                                                                                   
                                                                                                                                                                                                                      
EXPECTED = {'color': 'tomato', 'linestyle': '--', 'linewidth': 1.0, 'label': 'Expected'}                                                                                                                            
   
def style_ax(ax):                                                                                                                                                                                                   
    ax.set_xlabel('Time (s)', fontsize=9)
    ax.tick_params(labelsize=8)                                                                                                                                                                                     
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.6)                                                                                                                                                                         

# plot first figure with elevation (z(t)):                                                                                                                                                                                                                     
ax = axes[0, 0]
ax.plot(t, z, color='steelblue', linewidth=1.5, label=r'$z(t)$')                                                                                                                                                    
ax.axhline(5.0, **EXPECTED)                                                                                                                                                                                         
ax.set_ylabel('Elevation (m)', fontsize=9)                                                                                                                                                                           
ax.set_title('Elevation', fontsize=10)                                                                                                                                                                               
style_ax(ax)                                                                                                                                                                                                        

# plot second figure with vertical linear velocity:                                                                                                                                                                                                     
ax = axes[0, 1]                                                                                                                                                                                                     
ax.plot(t, vz, color='steelblue', linewidth=1.5, label=r'$v_z(t)$')
ax.axhline(0.0, **EXPECTED)                                                                                                                                                                                         
ax.set_ylabel(r'$v_z$ (m/s)', fontsize=9)                                                                                                                                                                    
ax.set_title('Vertical Linear Velocity', fontsize=10)
style_ax(ax)                                                                                                                                                                                                        

# plot third figure with euler angles for roll pitch and yaw:                                                                                                                                                                                                        
ax = axes[1, 0]                                                                                                                                                                                                     
ax.plot(t, phi, color='steelblue', linewidth=1.5, label=r'$\phi$ (roll)')                                                                                                                           
ax.plot(t, theta, color='darkorange', linewidth=1.5, label=r'$\theta$ (pitch)')                                                                                                                         
ax.plot(t, psi, color='seagreen', linewidth=1.5, label=r'$\psi$ (yaw)')                                                                                                                             
ax.axhline(0.0, **EXPECTED)                                                                                                                                                                                         
ax.set_ylabel('Euler Angles (rad)', fontsize=9)                                                                                                                                                                     
ax.set_title('Roll, Pitch, Yaw', fontsize=10)                                                                                                                                                                       
style_ax(ax)                                                                                                                                                                                                      

# last plot figure with angular velocity:                                                                                                                                                                                                                 
ax = axes[1, 1]                                                                                                                                                                                                   
ax.plot(t, omega_x, color='steelblue', linewidth=1.5, label=r'$\omega_x$')
ax.plot(t, omega_y, color='darkorange', linewidth=1.5, label=r'$\omega_y$')                                                                                                                                             
ax.plot(t, omemga_z, color='seagreen',   linewidth=1.5, label=r'$\omega_z$')                                                                                                                                             
ax.axhline(0.0, **EXPECTED)                                                                                                                                                                                         
ax.set_ylabel(r'Angular Velocity (rad/s)', fontsize=9)                                                                                                                                                       
ax.set_title('Body-Frame Angular Velocities', fontsize=10)                                                                                                                                                          
style_ax(ax)                                                                                                                                                                                                        
                                                                                                                                                                                                                      
plt.tight_layout()                                                                                                                                                                                                  
plt.savefig('figures/hover_test.png', dpi=150, bbox_inches='tight')                                                                                                                                                       
plt.show()