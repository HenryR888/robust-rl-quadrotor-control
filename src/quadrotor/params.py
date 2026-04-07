from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class Quadrotorparams:

    # physical quantities (we can adjust treat these as hyperparams, except g, these are just placeholder value for now): 

    m: float = 1.0 # mass of drone (kg) - calculated as follows: each MN2212 motor is approx. 54g. Thus, 216g total for motors. Mass of Nvidia Jetson Orin Nano = 176g. Battery = 311g. ESC Tekko32 50A = 13.8g. Propellers = 60g. FC + GPS = 30g. Frame = 300g. Wiring plus screws = 100g. TOTAL = 936.8g, we'll add 10% for MoE, so we have approx 1kg drone mass.  
    g: float = 9.81 # gravitational acceleration (m.s^(-2))

    # moments of inertia (these are hyperparameters which we shall need to tune within the model...the intuition is to keep these reasonably small as the moments of inertia describe how 'difficult' it would be to rotate the drone in that direction...there is obviously stability/maneuverability trade-off here): 
    # From A S Sanca et al. 2008: I_x = 2/5.M_sphere.r^2 + 2.l^2.M_rotor = I_y; I_z = 2/5.M_sphere.r^2 + 4.l^2.M_rotor. For the F450 proposed drone frame, we would have l = 0.159m, M_rotor = 0.069kg. M_sphere = M_total - 4M_rotor = 1kg - 0.276 = 0.724kg. Estimated thickness of drone = 160mm, thus r = 0.08m
    I_x: float = 0.005342
    I_y: float = 0.005342
    I_z: float = 0.008831

    # drone dimensions and torque-thrust ratio:

    l : float = 0.225 # arm length (m) from centre of mass
    k_d: float = 0.1 # torque-thrust ratio

    # maximum thrust in N, for a single MN2212 T-motor (equivalent to 1.8kg)
    max_rotor_thrust: float = 17.7

    @property
    def hover_thrust(self) -> float: # the amount of thrust per rotor needed to maintain hover...we will add the stochastic forces later. 
        return (self.m*self.g)/4.0
       
    # maximum roll and pitch torque produced form MN2212 T-motor:  
    @property
    def tau_xy_max(self) -> float:
        return 2*(self.l/np.sqrt(2))*self.max_rotor_thrust
    
    @property
    def tau_z_max(self) -> float:
        return 2*self.k_d*self.max_rotor_thrust