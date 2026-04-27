'''
Here we implement the Linear Quadratic Regulator Controller for the quadrotor:
'''

import numpy as np
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