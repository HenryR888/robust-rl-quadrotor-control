import numpy as np
from envs.hover_env import HoverEnv

def test_hover_env_constant_thrust():
    '''
    Here we test if the drone remains in the same initialised hover position after max_time_steps (set to 500).
    Our expectation is that the drone stays in exactly the same place, without moving, with effectively 0 position error. 
    '''

    env = HoverEnv() 
    obs, _ = env.reset()

    params = env.params
    hover_thrust_action = np.array([params.m * params.g, 0.0, 0.0, 0.0])

    print(f"Hover thrust: {hover_thrust_action[0]:.4f}N")
    print(f"Drone is initialised at: position={obs[0:3]}, linear_velocity={obs[3:6]}")

    max_pos_error = 0.0 # initialise max_pos_error here

    for step in range(env.max_steps):

        obs, reward, terminated, truncated, _ = env.step(hover_thrust_action) # we get the state observation, reward, is_terminated BOOL, and is_truncated BOOL from the step function in hover_env
        max_pos_error = max(max_pos_error, np.linalg.norm(obs[0:3]-env.target))

        if terminated: 
            print(f"ERROR: terminated too early at time step {step+1}")
            print(f"position={obs[0:3]}, phi={obs[6]:.4f}, theta={obs[7]:.4f}")
            break
    else: 
        print(f"Completed {env.max_steps} steps")
        print(f"Maximum Position Error: {max_pos_error:2e} m")
        print(f"Final Position: {obs[0:3]}, Final Linear Velocity: {obs[3:6]}")
        print(f"Final reward: {reward:.6f}")
        print()
        print("SUCCESS" if max_pos_error<1e-6 else f"DRIFT TOO LARGE...DEBUG NECESSARY: {max_pos_error:.2e}")
        
test_hover_env_constant_thrust()
                      