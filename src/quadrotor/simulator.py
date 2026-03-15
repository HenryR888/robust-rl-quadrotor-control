'''
In this file, we shall implement the simulator to check that our drone dynamics behave as we expect. 
The way in which we confirm this is by intergrating our 12 ODEs forward in time, over a time horizon and check that the output is as we expected. 
'''

import numpy as np
from scipy.integrate import solve_ivp
from quadrotor.dynamics import f 
from quadrotor.params import Quadrotorparams


def simulation(x_0: np.array, u: np.array, t_span: tuple, params:Quadrotorparams, t_eval: np.array = None):

    sol = solve_ivp(
        lambda t, x: f(x,u,params),
        t_span,
        x_0,
        method='RK45',
        t_eval=t_eval
    )
    return sol 
