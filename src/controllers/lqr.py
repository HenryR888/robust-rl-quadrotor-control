'''
Here we implement the Linear Quadratic Regulator Controller for the quadrotor:
'''

import numpy as np
from scipy.linalg import solve_continuous_are
from quadrotor.params import Quadrotorparams

def _build_AB(params: Quadrotorparams):
    '''
    Here we build our Jacobian matrices A (12x12) and B (12x4) for our linearised system about the hover equilibrium position.
    '''
    g, m = params.g, params.m
    I_x, I_y, I_z = params.I_x, params.I_y, params.I_z

    # we create our A matrix in accordance with the derived Jacobian matrix A in the docs: 
    A = np.zeros((12,12))
    A[0,3]=1.0
    A[1,4]=1.0
    A[2,5] = 1.0
    A[3,7]=g
    A[4,6] = -g
    A[6,9] = 1.0
    A[7,10] = 1.0
    A[8,11] = 1.0

    # then we create our B matrix in accordance with the derived Jacobian matrix B in the docs: 
    B = np.zeros((12,4))
    B[5,0] = 1.0/m
    B[9,1] = 1.0/I_x
    B[10,2] = 1.0/I_y
    B[11,3] = 1.0/I_z

    return A,B

class LQRController:

    def __init__(self,params: Quadrotorparams):
        self.params = params

        A, B = _build_AB(params)

        Q = np.diag([ # we set Q according to Bryson's rule, which means that Q_ii = 1/xtilde_i_max^2.
            1.0, 1.0, 4.0, # here we choose max x error to be 1m, y error to be 1m and z error to be 0.5m...we can change these a bit later, I want to be lenient for initial tuning.
            1.0, 1.0, 4.0, # x error to be 1m/s, y error to be 1m/s
            33.0, 33.0, 8.0, # phi, and theta are more strict, we want them to be within 10deg of amount...yaw we can be more lenient for initial tuning, allowing it to be 20deg off 
            4.0, 4.0, 4.0, # angular velocity must be within 0.5rad/s error for all of x,y,z directions
        ])

        R = np.diag([
            0.11, # deviation in thrust is 3N
            1.0, # max deviation in the roll torque is 1Nm, which is approx. 20% of max roll torque
            1.0,
            2.0, # max deviation in yaw torque is 0.708Nm, which is also approx. 20% of max yaw torque
        ])

        # here we solve the continuous algebraic Riccati Equation (CARE): 
        P = solve_continuous_are(A,B,Q,R)

        # then we compute our feedback controller gain matrix, K = R^-1.B'.P:
        self.K = np.linalg.inv(R)@ B.T @ P