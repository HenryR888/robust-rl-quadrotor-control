'''
We test that the yaw dynamics work correctly here. We apply equal thrust to rotors 1 and 3 or 2 and 4, and see that the yaw angle changes with time, and yaw angular velocity changes with time. 
For a counter-clockwise (CCW) yaw, we set u_2 = u_4, with u_1 = u_3 and u_2 > u_1 to accomplish the CCW yaw. We should expect that the omega_z and psi are the two values that change, the position of
quadrotor should only experience slight changes in the z-direction. 
'''

import sys
sys.path.append('src')

import numpy as np 
import matplotlib.pyplot as plt
from quadrotor.params import Quadrotorparams
from quadrotor.simulator import simulation

# fetch our parameters: 
params = Quadrotorparams()

# start at some vertical position 'x' m, perfectly straight and level:
x_0 = np.array([0,0,5, # corresponds to position ICs
                0,0,0, # corresponds to velocity ICs
                0,0,0, # corresponds to euler angle ICs
                0,0,0]) # corresponds to angular velocity ICs

# apply yaw thrust accordingly (CCW yaw is u_2=u_4 and u_1=u_3, with u_2>u_1):
h = params.hover_thrust
u = np.array([0.9*h,1.1*h,0.9*h,1.1*h])
#u = np.array([1.1*h,0.9*h,1.1*h,0.9*h]) #CLOCKWISE yaw

# setup parameters to be fed into simulation function: 
t_span = (0,100)
t_eval = np.linspace(0,100,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we can plot for z, psi, and omega_z), for z it should start at zero and then increase or decrease; psi should slowly start peeling away from 0 and then move away from 0;
# in this case omega_z must be a straight line, since tau_z is constant, and omega_x, omega_y are 0: 
plt.plot(sol.t,sol.y[8])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Angle (rad)')
plt.title('Pure Pitch Test')
plt.grid(True)
plt.show()
