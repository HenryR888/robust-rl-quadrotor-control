'''
In this file, we shall implement the simulator to check that our drone dynamics behave as we expect. 
The way in which we confirm this is by intergrating our 12 ODEs forward in time, over a time horizon and check that the output is as we expected. 
'''

import numpy as np
from scipy.integrate import solve_ivp
from quadrotor.dynamics import f 
from quadrotor.params import Quadrotorparams


def simulation(x_0: np.ndarray, u: np.ndarray, t_span: tuple, params:Quadrotorparams, t_eval: np.ndarray = None):

    sol = solve_ivp(
        lambda t, x: f(x,u,params),
        t_span,
        x_0,
        method='RK45',
        t_eval=t_eval
    )
    return sol 

def rk4_method(x0: np.ndarray, u: np.ndarray, dt: float, params: Quadrotorparams) -> np.ndarray: 
    k1 = f(x0,u, params)
    k2 = f(x0 + 0.5*dt*k1, u, params)
    k3 = f(x0 + 0.5*dt*k2, u, params)
    k4 = f(x0 + dt*k3, u, params)
    return x0 + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)

