'''
Here we create the hover environment for the quadrotor drone. This is compatible with the Gymnasium interface,
which shall allow our PID, LQR and PPO controllers to be compared within the same environment for fair comparison. 

The process works as follows: 
The agent (controller) observes the full 12-state and outputs [T,tau_x, tau_y, tau_z]. The episode target here is for the drone to hover at a fixed position. 
'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces # used to define set of valid actions and observations
from quadrotor.params import Quadrotorparams
from quadrotor.dynamics import f, thrust_matrix
from quadrotor.simulator import rk4_method

class HoverEnv(gym.Env):

    def __init__(self, target: np.ndarray = np.array([0.0,0.0,1.0])):
        super().__init__()
        self.params = Quadrotorparams()
        self.target = target
        self.dt = 0.01 # this is the amount of real time that passes per time step (this corresponds to 10^-3s...!Note: May need adjusting)
        self.max_steps = 500 # thus, the episode length will be 5s of hover time (I'm setting up a baseline where the drone is initiated in its hover position, and we check that it stays there)
        self.step_count = 0 # initiate step count at 0, and then we update it in step() function
        self.state = None 

        # weights for our quadratic reward function !Note: (still need to be tuned). 
        # We can start with the following reward function for PPO: r_t = -(w_pos||p-p*||^2 + w_vel||v||^2 + w_angle||phi^2 + theta^2|| + w_omega||omega||^2 + w_eff||u-u_hov||^2)
        # the reasoning for this reward function is it follows a similar mechanism to the LQR (Quadratic) cost function, which is a quadratic cost function dependent on performance and control effort. I am trying to minimise variation here so we can
        # have a statistically fair comparison between control methods (LQR, PPO) in their ability for robust stabilisation of the drone.
        self.w_pos = 1.0
        self.w_vel = 0.5
        self.w_angle = 0.5
        self.w_omega = 0.5
        self.w_eff = 0.25

        # Define action space, as per gymnasium spaces API: 
        action_low = np.array([0.0,-self.params.tau_xy_max,-self.params.tau_xy_max,-self.params.tau_z_max])
        action_high = np.array([4.0*self.params.max_rotor_thrust,self.params.tau_xy_max, self.params.tau_xy_max, self.params.tau_z_max])
        self.action_space = spaces.Box(low=action_low, high=action_high, dtype=np.float64) # numpy uses float64 as default, but when running PPO we shall cast to float32 for more efficient training

        # Define observation space, as per the gymnasium API (observations aren't restricted...we want to be able to observe any value that the simulator yields): 
        obs_low = -np.inf* np.ones(12)
        obs_high = np.inf * np.ones(12)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float64)


        # invert the thrust matrix to obtain the thrust input vector for the motors in the step function: 

        self.Thrust_matrix = thrust_matrix(self.params)
        self.Thrust_matrix_inverse = np.linalg.inv(self.Thrust_matrix)

    # Define reset as per Gymnasium API: 
    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # !Note: we shall set a seed here later once we start at randomly initialised positions to ensure reproducibility
        self.state = np.array([
            0.0, 0.0, 1.0, # start position at the hover position...!Note: this is just for now, we will ultimately have drone start at some random position and fly to target position and then hover there
            0.0, 0.0, 0.0, # vx,vy, vz
            0.0, 0.0, 0.0, # phi, theta, psi 
            0.0, 0.0, 0.0 # omegax, omegay, omegaz
        ])
        self.step_count = 0
        return self.state.copy(), {}
        
    def step(self, thrust_tor_vec: np.ndarray):

        u = self.Thrust_matrix_inverse @ thrust_tor_vec # obtain input thrust vector to be passed back into function f from dynamics, which expects this u vector
        u = np.clip(u,0.0,None) # ensure that u is greater than or eq to 0, otherwise return 0...since it cannot have negative thrust

        self.state = rk4_method(self.state, u, self.dt, self.params)
        self.step_count += 1

        
    def _compute_reward(self, thrust_tor_vec: np.ndarray):
        pos = self.state[0:3]
        vel = self.state[3:6]
        angle = self.state[6:8]
        omega = self.state[9:12]

        hover_thrust = np.array([self.params.hover_thrust*4, 0.0, 0.0, 0.0])
        thrust_err = thrust_tor_vec - hover_thrust
        reward = (-self.w_pos * np.dot(pos-self.target, pos-self.target)
                  - self.w_vel * np.dot(vel, vel)
                  - self.w_angle * np.dot(angle, angle)
                  - self.w_omega * np.dot(omega, omega)
                  - self.w_eff * np.dot(thrust_err, thrust_err))
        return reward