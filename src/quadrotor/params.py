from dataclasses import dataclass

@dataclass(frozen=True)
class Quadrotorparams:

    # physical quantities (we can adjust treat these as hyperparams, except g, these are just placeholder value for now): 

    m: float = 0.5 # mass of drone (kg)
    g: float = 9.81 # gravitational acceleration (m.s^(-2))

    # moments of inertia (these are hyperparameters which we shall need to tune within the model...the intuition is to keep these reasonably small as the moments of inertia describe how 'difficult' it would be to rotate the drone in that direction...there is obviously stability/maneuverability trade-off here): 

    I_x: float = 0.1
    I_y: float = 0.1
    I_z: float = 0.1

    # drone dimensions and torque-thrust ratio:

    l : float = 0.2 # arm length (m) from centre of mass
    k_d: float = 0.1 # torque-thrust ratio

    @property
    def hover_thrust(self) -> float: # the amount of thrust per rotor needed to maintain hover...we will add the stochastic forces later. 
        return (self.m*self.g)/4.0
