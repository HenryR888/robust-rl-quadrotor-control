'''
Here we create the hover environment for the quadrotor drone. This is compatible with the Gymnasium interface,
which shall allow our PID, LQR and PPO controllers to be compared within the same environment for fair comparison. 

The process works as follows: 
The agent (controller) observes the full 12-state and outputs [T,tau_x, tau_y, tau_z]. The episode target here is for the drone to hover at a fixed position. 

We also add the ability to simulate stochastic wind disturbances using the following AR(1) process: 

F_t = alpha.F_{t-1} + (1-alpha).F_mean + (k.F_magnitude).sqrt(1-alpha^2).epsilon_t, where epsilon_t ~ N(0,1), and alpha controls the temporal correlation`
- (k.F_magnitude) represents the variability of the wind about the mean wind force.
- we note that: F_mean = F_magnitude * [cos(gamma_w)cos(beta_w), cos(gamma_w)sin(beta_w), sin(gamma_w)]

Further, we note that beta_w ~ U(0,2pi) which represents the horizontal wind angle in spherical coords, which is drawn per seed.
Moreover, gamma_w ~ U[-gamma_max, gamma_max] represents the wind elevation angle in spherical coords, which is drawn per seed. 

'''

import numpy as np
import gymnasium as gym
from gymnasium import spaces # used to define set of valid actions and observations
from quadrotor.params import Quadrotorparams
from quadrotor.dynamics import f, thrust_matrix
from quadrotor.simulator import rk4_method

class HoverEnv(gym.Env):

    def __init__(self, target: np.ndarray = np.array([0.0,0.0,1.0]),
                wind_magnitude: float = 0.0,
                k: float=0.15,
                alpha: float = 0.98,
                gamma_max: float = np.pi/12):
        super().__init__()
        self.params = Quadrotorparams()
        self.target = target
        self.dt = 0.01 # this is the amount of real time that passes per time step (this corresponds to 10^-2s...!Note: May need adjusting)
        self.max_steps = 5000 # thus, the episode length will be 5s of hover time (I'm setting up a baseline where the drone is initiated in its hover position, and we check that it stays there)...!Note: adjust as necessary for tuning
        self.step_count = 0 # initiate step count at 0, and then we update it in step() function
        self.state = None 
        self.wind_magnitude = wind_magnitude
        self.k = k
        self.alpha = alpha
        self.gamma_max = gamma_max
        self.F_mean = np.zeros(3)
        self.F_wind = np.zeros(3)

        # weights for our quadratic reward function !Note: (still need to be tuned). 
        # We can start with the following reward function for PPO: r_t = 0.1 -(w_pos||p-p*||^2 + w_vel||v||^2 + w_roll_pitch||phi^2 + theta^2|| + w_yaw||psi^2|| + w_omega||omega||^2 + w_eff||u-u_hov||^2)
        # the reasoning for this reward function is it follows a similar mechanism to the LQR (Quadratic) cost function, which is a quadratic cost function dependent on performance and control effort. I am trying to minimise variation here so we can
        # have a statistically fair comparison between control methods (LQR, PPO) in their ability for robust stabilisation of the drone. We also add the 0.1 initially as a survival factor for early episodes of training
        self.w_pos = 1.0
        self.w_vel = 0.5
        # we remove the w_roll_pitch to maintain consistent reward design throughout the entire experiment, since the quadrotor may tilt into the wind to counter wind forces, and thus do 
        # not want to penalise the controller for tilting into the wind for stochastic wind disturbances. Moreover, we want to maintain consistent reward design to remove confounding between baseline and disturbance conditions
        #self.w_roll_pitch = 2.0
        self.w_yaw = 2.0
        self.w_omega = 2.0
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
        # we randomise the reset position so the drone starts off with non-zero error: 
        pos = self.np_random.uniform(-0.3, 0.3, size=3)+np.array([0.0,0.0,1.0]) # here we start the drone around position [0,0,1] each reset
        vel = self.np_random.uniform(-0.1,0.1,size=3)
        angle = self.np_random.uniform(-0.1,0.1, size=3)
        self.state = np.concatenate([pos, vel, angle, np.zeros(3)])
        self.step_count =0
        beta_w = self.np_random.uniform(0.0, 2.0*np.pi)
        gamma_w = self.np_random.uniform(-self.gamma_max, self.gamma_max)
        self.F_mean = self.wind_magnitude*np.array([
            np.cos(gamma_w)*np.cos(beta_w),
            np.cos(gamma_w)*np.sin(beta_w),
            np.sin(gamma_w)
        ])
        self.F_wind = self.F_mean.copy() # here we initialise the wind force as the average wind speed recorded 
        return self.state.copy(), {}

        #self.state = np.array([
            #0.0, 0.0, 1.0, # start position at the hover position...!Note: this is just for now, we will ultimately have drone start at some random position and fly to target position and then hover there
            #0.0, 0.0, 0.0, # vx,vy, vz
            #0.0, 0.0, 0.0, # phi, theta, psi 
            #0.0, 0.0, 0.0 # omegax, omegay, omegaz
        #])
       # self.step_count = 0
       # return self.state.copy(), {}
        
    def step(self, thrust_tor_vec: np.ndarray):

        u = self.Thrust_matrix_inverse @ thrust_tor_vec # obtain input thrust vector to be passed back into function f from dynamics, which expects this u vector
        u = np.clip(u,0.0,None) # ensure that u is greater than or eq to 0, otherwise return 0...since it cannot have negative thrust

        self.state = rk4_method(self.state, u, self.dt, self.params)
        eps = self.np_random.standard_normal(3)
        self.F_wind = self.alpha * self.F_wind + (1.0-self.alpha)*self.F_mean + (self.k*self.wind_magnitude*np.sqrt(1.0-self.alpha**2))*eps
        # we then add the change in velocity (deltav) to our velocity values, which is given by a.dt = (F/m).dt. We note that adding the change in velocity here introduces 
        self.state[3:6] += (self.F_wind/self.params.m)*self.dt 
        self.step_count += 1

        reward = self._compute_reward(thrust_tor_vec) # compute the reward as per below
        terminated = self._is_terminated() # check whether episode is terminated 
        truncated = self.step_count >= self.max_steps # check whether episode ends based on truncation
        # here we add a -50 reward to the drone, to signal that crashing or flipping cannot occur. 
        if terminated:
            reward -= 50.0

        return self.state.copy(), reward, terminated, truncated, {}

    
    
    def _compute_reward(self, thrust_tor_vec: np.ndarray):
        '''
        - reward function for PPO: r_t = 0.1 -(w_pos||p-p*||^2 + w_vel||v||^2 + w_roll_pitch||phi^2 + theta^2||+ w_yaw||psi^2|| + w_omega||omega||^2 + w_eff||u-u_hov||^2)
        - the reasoning for this reward function is it follows a similar mechanism to the LQR (Quadratic) cost function, which is a quadratic cost function dependent on performance and 
          control effort. I am trying to minimise variation here so we can
          have a statistically fair comparison between control methods (LQR, PPO) in their ability for robust stabilisation of the drone.
        - note here that we've added a 0.1 value to signal positive reward for staying alive (not terminating) i.e. not crashing or flipping
        '''
        
        pos = self.state[0:3]
        vel = self.state[3:6]
        roll_pitch = self.state[6:8]
        yaw = self.state[8]
        omega = self.state[9:12]

        hover_thrust = np.array([self.params.hover_thrust*4, 0.0, 0.0, 0.0])
        thrust_err = thrust_tor_vec - hover_thrust
        reward = (0.1
                  -self.w_pos * np.dot(pos-self.target, pos-self.target)
                  - self.w_vel * np.dot(vel, vel)
                  - self.w_yaw*yaw**2
                  - self.w_omega * np.dot(omega, omega)
                  - self.w_eff * np.dot(thrust_err, thrust_err))
        return reward
    
    def _is_terminated(self):
        z = self.state[2]
        phi = self.state[6]
        theta = self.state[7]
        crashed = z <= 0.0
        flipped = abs(phi) > np.pi/3 or abs(theta) > np.pi/3 # for this 'flipped' quantity I need to add a calculated restriction inside of params...which would be given by total vertical thrust>= total downward force. The total vertical thrust = thrust_max.cos(phi).cos(theta)...based on a quick calc. from params, we  have that 60deg = pi/3 rad is a decent estimate
        return bool(crashed or flipped)