'''
Here we perform a hover test. We initiate the hover at 'x' m above the ground (we can tweak this height of course), and verify that under no stochastic disturbances,
(addition of this will come later), the height should remain constant. Additionally, the attitude of the drone (in all roll, pitch and yaw directions) should not change...i.e. they should remain at 0.
'''

import sys
sys.path.append('src')

import numpy as np 
import matplotlib.pyplot as plt
from quadrotor.params import Quadrotorparams
from quadrotor.simulator import simulation

# fetch our parameters: 
params = Quadrotorparams()

# start at hover position, exactly 'x' m up, no yaw, roll or pitch (straight and level):
x_0 = np.array([0,0,5, # corresponds to position ICs
                0,0,0, # corresponds to velocity ICs
                0,0,0, # corresponds to euler angle ICs
                0,0,0]) # corresponds to angular velocity ICs

# check that applying exact amount of hover thrust keeps quadrotor level: 
u = np.array([params.hover_thrust]*4)

# setup parameters to be fed into simulation function: 
t_span = (0,50)
t_eval = np.linspace(0,50,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we should expect a flat horizontal line): 
plt.plot(sol.t,sol.y[2])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Elevation (m)')
plt.title('Hover Test')
plt.grid(True)
plt.show()
