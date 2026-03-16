'''
We test that the pitch dynamics work correctly here. We apply equal thrust to rotors 1 and 2 or 3 and 4, and see that the pitch angle changes with time, and pitch angular velocity changes with time. 
For a forward (positive) pitch, we set u_3 = u_4, with u_1 = u_2 and u_3 > u_1 to accomplish the pitch. We should expect that the omega_y and theta are the two values that change, as well as z (if not enough
thrust is applied to keep drone at constant elevation), and x should change too (remember by the right hand rule that forward is positive). 
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

# apply pitch thrust accordingly (FORWARD pitch is when u_3=u_4 & u_1=u_2, but u_3>u_1):
h = params.hover_thrust
#u = np.array([0.9*h,0.9*h,1.1*h,1.1*h])
u = np.array([1.1*h,1.1*h,0.9*h,0.9*h]) #BACKWARD pitch

# setup parameters to be fed into simulation function: 
t_span = (0,3)
t_eval = np.linspace(0,3,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we can plot for x, z, theta, and omega_y), for x it should start at zero and then increase or decrease; z the same; theta should slowly start peeling away from 0 and then move away from 0;
# in this case omega_y must be a straight line, since tau_y is constant, and omega_x, omega_z are 0: 
plt.plot(sol.t,sol.y[7])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Angle (rad)')
plt.title('Pure Pitch Test')
plt.grid(True)
plt.show()
