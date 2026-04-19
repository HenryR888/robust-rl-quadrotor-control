'''
Here we implement a Cascaded PID controller. 

We split the PID controller into two loops: We have:
- an outer loop, which takes position error and outputs desired roll/pitch angles + total thrust
- and an inner loop which takes the attitude error and outputs the roll/pitch/yaw torques

The action output for the quadrotor shall then match the action space: [T, tau_x, tau_y, tau_z]
'''

import numpy as np
from quadrotor.params import Quadrotorparams

class PIDGains:
    def __init__(self, kp, ki, kd, integral_limit=np.inf):
        # 3 scalar gains per axis: 
        self.kp = kp # proportional gain
        self.ki = ki # integral gain
        self.kd = kd # derivative gain
        self.integral_limit = integral_limit # clamp for anti-windup 

class PIDState:
    def __init__(self):
        # we instantiate the integral and prev_error, as we require these terms for adding up the integral values and computing the approximate derivative term de/dt = (ek-e_{k-1})/dt
        self.integral =0.0
        self.prev_error = 0.0
        self._first_call = True

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self._first_call = True
    
class CascadedPIDController:

    def __init__(self, params: Quadrotorparams):
        self.params = params 

        # z-axis gains:
        self.gains_z = PIDGains(kp=2.0,ki=0.0, kd=1.5, integral_limit=0.5) # TUNING COMPLETE

        # x and y axis position gains:
        self.gains_x = PIDGains(kp=0.003,ki=0.0, kd=0.05, integral_limit=0.0) # TUNING COMPLETE
        self.gains_y = PIDGains(kp=0.003,ki=0.0, kd=0.05, integral_limit=0.0) # TUNING COMPLETE

        # even though within hover_env we set a limit of pi/4 rads before drone flips...we want some margin to prevent controller pushing drone to limit, and stochastic disturbance arises pushing drone into flip. 
        self.max_tilt = np.pi/6

        # Attitude gains for inner control loop: 
        self.gains_phi = PIDGains(kp=0.005,ki=0.0, kd=0.01, integral_limit=0.5) # TUNING COMPLETE
        self.gains_theta = PIDGains(kp=0.003,ki=0.0, kd=0.01, integral_limit=0.5) # TUNING COMPLETE
        self.gains_psi = PIDGains(kp=0.02,ki=0.0, kd=0.06, integral_limit=0.5) # TUNING COMPLETE

        # we have 6 PID controllers, one controlling each channel. Thus, we need each controller to have separate integral and prev_error values (i.e. PID states): 
        self._z = PIDState()
        self._x = PIDState()
        self._y = PIDState()
        self._phi = PIDState()
        self._theta = PIDState()
        self._psi = PIDState()

    def reset(self): # reset all the integrals and prev errors for 6 controllers: 
        for s in [self._z, self._x, self._y, self._phi, self._theta, self._psi]:
            s.reset()

    def _pid_step(self, error: float, state: PIDState, gains:PIDGains, dt:float) -> float:
        state.integral = np.clip(
            state.integral + error*dt, # here we compute the integral term by simply adding prev_error and multipling by small change in time. 
            -gains.integral_limit, # we clip for anti-windup, to prevent drone overshooting if control output saturates. 
            gains.integral_limit
        )

        if state._first_call:
            derivative = 0.0
            state._first_call = False
        else:
            derivative = (error - state.prev_error)/dt # approximate derivative term numerically by: e_t - e_{t-1}/dt
        state.prev_error = error
        return  (gains.kp*error) + (gains.ki*state.integral) + (gains.kd*derivative) # output control variable 
    
    def compute_action(self, obs:np.ndarray, target: np.ndarray, dt: float) -> np.ndarray:
        '''
        obs: full 12-states [x,y,z,vx,vy,vz,phi,theta,psi,omegax,omegay,omegaz]
        target/setpoint: desired position [x*,y*,z*]
        dt: timestep (in seconds)
        returns: [T,tau_x,tau_y, tau_z]
        '''

        x, y, z = obs[0], obs[1], obs[2]
        phi, theta, psi = obs[6], obs[7], obs[8]

        # OUTER LOOP:

        # control loop for Elevation, z. Our process variable is z, which we want to reach our target/setpoint. This is done by manipulation of our control variable, T_correction, thrust input:

        e_z = target[2] - z
        T_correction = self._pid_step(e_z, self._z, self.gains_z, dt) # recall that our control variable is T, and thus our output from the _pid_step is the amount of thrust change to the hover thrust
        T = self.params.m * self.params.g + T_correction
        T = np.clip(T, 0.0, 4.0*self.params.max_rotor_thrust)

        # PID control loop on position to output desired tilt angle. Here we have control x, and y positions as our process variable. But to move in those directions, we need to pitch and roll respectively. Thus, the control variables are theta, and phi
        e_x = target[0]-x
        e_y = target[1] - y
        raw_theta_des = self._pid_step(e_x, self._x, self.gains_x, dt) 
        raw_phi_des = -self._pid_step(e_y, self._y, self.gains_y, dt)

        # clip to max tilt lims as set above: 
        theta_des = np.clip(raw_theta_des, -self.max_tilt, self.max_tilt)
        phi_des = np.clip(raw_phi_des, -self.max_tilt, self.max_tilt)

        psi_des = 0.0 # for the desired hover, we always want to face forward...we shall change this should we want the drone to face in another direction


        # INNER LOOP: 

        # here our inner loop is controlling the desired attitude of the drone. Our process variables are phi, theta and psi respectively, and our targets/setpoints come from our outerloop, which tell us how much we need to pitch/roll in order to move to the desired position.
        # our control variables here are our torques, which allow us to change the process variables. 
        e_phi = phi_des - phi
        e_theta = theta_des - theta
        e_psi = psi_des - psi

        raw_tau_x = self._pid_step(e_phi, self._phi, self.gains_phi, dt)
        raw_tau_y = self._pid_step(e_theta, self._theta, self.gains_theta, dt)
        raw_tau_z = self._pid_step(e_psi, self._psi, self.gains_psi, dt)

        # clip to physical lims: 
        tau_x = np.clip(raw_tau_x, -self.params.tau_xy_max, self.params.tau_xy_max)
        tau_y = np.clip(raw_tau_y, -self.params.tau_xy_max, self.params.tau_xy_max)
        tau_z = np.clip(raw_tau_z, -self.params.tau_z_max, self.params.tau_z_max)

        return np.array([T, tau_x, tau_y, tau_z])



        
