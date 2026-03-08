'''
this file, we shall define our rotation matrix R(phi, theta, psi);

Thrust matrix Cal_T, which will allow us to calculate a vector for: total Thrust, roll torque, pitch torque and yaw torque: [T, tau_x, tau_y, tau_z];

and assemble our 12 ODEs
'''

import numpy as np
from quadrotor.params import Quadrotorparams

def rotation_matrix(phi, theta, psi):

    cp, sp = np.cos(phi), np.sin(phi) # cos(phi), and sin(phi)
    ct, st = np.cos(theta), np.sin(theta) # cos(theta) and sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi) # cos(psi) and sin(psi)

    # this is the same array we derived in order derivations pdf
    return np.array([
        [cpsi*ct, cpsi*st*sp - spsi*cp, cpsi*st*cp + spsi*sp],
        [spsi*ct, spsi*st*sp + cpsi*cp, spsi*st*cp - cpsi*sp],
        [-st, ct*sp, ct*cp]
    ])

# thrust matrix to produce the thrust and torque vector from set of equations, as from the derivations and thrust mapping pdf:
# T = u_1 + u_2 + u_3 + u_4
# tau_x = (l/sqrt(2)).(-u_1 + u_2 + u_3 - u_4)
# tau_y = (l/sqrt(2)).(-u_1 - u_2 + u_3 + u_4)
# tau_z = k_d.(-u_1 + u_2 - u_3 + u_4)

def thrust_matrix(params):
    s = params.l/np.sqrt(2)
    k_d = params.k_d

    return np.array([
        [1, 1, 1, 1],
        [-s, s, s, -s],
        [-s, -s, s, s],
        [-k_d, k_d, -k_d, k_d]
    ])

# assemble all 12 ODEs: 

def f(x, u, params):

    v_x, v_y, v_z = x[3], x[4], x[5]
    phi, theta, psi = x[6], x[7], x[8]
    w_x, w_y, w_z = x[9], x[10], x[11]

    # Compute thrust and torque vector:

    T, tau_x, tau_y, tau_z = thrust_matrix(params) @ u

    # assemble the xdot, ydot, and zdot vector: 

    pos_dot = np.array([v_x, v_y, v_z])

    # Translational kinematics to assemble velocity_dot vector: 

    R = rotation_matrix(phi, theta, psi)
    thrust_force = np.array([0.0, 0.0, T])
    vel_dot = np.array([0.0, 0.0, -params.g]) + (1.0/params.m)* R @ thrust_force

    # Rotational kinematics to assemble eta_dot vector:

    cp, sp = np.cos(phi), np.sin(phi)
    ct, st = np.cos(theta), np.sin(theta)

    W = np.array([
        [1, 0, -st],
        [0, cp, sp*ct],
        [0, -sp, cp*ct]
    ])

    # invert matrix...(might be worth actually analytically inverting to be more compute efficient)
    W_inv = np.linalg.inv(W)

    eta_dot = W_inv @ np.array([w_x, w_y, w_z])

    # Rotational dynamics to assemble omega_dot vector: 

    I_x, I_y, I_z = params.I_x, params.I_y, params.I_z

    omega_dot = np.array([
        (tau_x - (I_z-I_y)*w_y*w_z)/I_x,
        (tau_y - (I_x - I_z)*w_x*w_z)/I_y,
        (tau_z - (I_y-I_x)*w_x*w_y)/I_z
    ])

    # Finally, assemble the full vector of RHS part of 12 ODEs:

    return np.concatenate([pos_dot, vel_dot, eta_dot, omega_dot])

