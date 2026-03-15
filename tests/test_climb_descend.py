'''
Here we perform a climb and descend test. We let the quadrotor start climbing at 'x' m, and set the thrust to be greater than the hover thrust. For descent, we set the thrust to be lower than
hover thrust. We expect to see the drone's elevation increase, when climbing. And when descending, the drone's elevation should decrease at a rate much less than under freefall. 
Moreover, the more we increase the thrust, the steeper the climb gradient should be. And for descent, the more we reduce the thrust, the steeper (more negative) the descent gradient should be.
Additionally, the attitude of the drone (in all roll, pitch and yaw directions) should not change...i.e. they should remain at 0.
'''

import sys
sys.path.append('src')

import numpy as np 
import matplotlib.pyplot as plt
from quadrotor.params import Quadrotorparams
from quadrotor.simulator import simulation

# fetch our parameters: 
params = Quadrotorparams()

# start at position 'x' m up, no yaw, roll or pitch (straight and level):
x_0 = np.array([0,0,5, # corresponds to position ICs
                0,0,0, # corresponds to velocity ICs
                0,0,0, # corresponds to euler angle ICs
                0,0,0]) # corresponds to angular velocity ICs

# set thrust to be greater than hover thrust: 
u = np.array([0.5*params.hover_thrust]*4)

# setup parameters to be fed into simulation function: 
t_span = (0,5)
t_eval = np.linspace(0,5,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we should expect an increasing exponential for climb and decreasing exponential for descent): 
plt.plot(sol.t,sol.y[2])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Elevation (m)')
plt.title('Climb Test')
plt.grid(True)
plt.show()
