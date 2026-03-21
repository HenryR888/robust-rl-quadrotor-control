'''
Here we perform a freefall test. We release the quadrotor from some height without any thrust applied, and we expect the quadrotor to fall to the ground. 
Additionally, the attitude of the drone (in all roll, pitch and yaw directions) should not change...i.e. they should remain at 0.
'''

import numpy as np 
import matplotlib.pyplot as plt
from quadrotor.params import Quadrotorparams
from quadrotor.simulator import simulation

# fetch our parameters: 
params = Quadrotorparams()

# release drone at exactly 'x' m up, no yaw, roll or pitch (straight and level):
x_0 = np.array([0,0,5, # corresponds to position ICs
                0,0,0, # corresponds to velocity ICs
                0,0,0, # corresponds to euler angle ICs
                0,0,0]) # corresponds to angular velocity ICs

# rotors switched off: 
u = np.array([0.0,0.0,0.0,0.0])

# setup parameters to be fed into simulation function: 
t_span = (0,1)
t_eval = np.linspace(0,1,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we should expect a negative gradient parabola): 
plt.plot(sol.t,sol.y[2])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Elevation (m)')
plt.title('Free Fall Test')
plt.grid(True)
plt.show()
