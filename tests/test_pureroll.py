'''
We test that the roll dynamics work correctly here. We apply equal thrust to rotors 1 and 4 or 2 and 3, and see that the roll angle changes with time, and roll angular velocity changes with time. 
For a left roll, we set u_1 = u_4, with u_2 = u_3 and u_1 > u_2 to accomplish the roll. We should expect that the omega_x and phi are the two values that change, as well as z (if not enough
thrust is applied to keep drone at constant elevation), and y should change too (remember by the right hand rule that positive y is to the left). 
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

# apply roll thrust accordingly (LEFT roll is when u_1=u_4 & u_2=u_3, but u_1>u_2):
h = params.hover_thrust
u = np.array([1.1*h,0.9*h,0.9*h,1.1*h])
#u = np.array([0.9*h,1.1*h,1.1*h,0.9*h]) #RIGHT roll

# setup parameters to be fed into simulation function: 
t_span = (0,5)
t_eval = np.linspace(0,5,200)

# fetch the solution from the simulator:
sol = simulation(x_0, u, t_span, params, t_eval)

# plot the result (we can plot for y, z, phi, and omega_x), for y it should start at zero and then increase or decrease; z the same; phi should slowly start peeling away from 0 and then move away from 0;
# in this case omega_x must be a straight line, since tau_x is constant, and omega_y, omega_z are 0: 
plt.plot(sol.t,sol.y[6])
plt.xlabel('Time Elapsed (s)')
plt.ylabel('Angle (rad)')
plt.title('Pure Roll Test')
plt.grid(True)
plt.show()
