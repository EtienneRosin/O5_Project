"""Configuration validation."""
import numpy as np
from typing import Optional

from pss.utils import Domain
from pss.waves.base import InitialCondition
from pss.potentials.base import Potential
from pss.potentials import Wall
from pss.waves import GaussianWavePacket

def validate_configuration(
    spatial_domain: Domain,
    temporal_domain: Domain,
    initial_condition: InitialCondition,
    potential: Optional[Potential] = None
) -> bool:
    """Validate simulation parameters."""
    
    L = spatial_domain.length.min()
    ds = np.linalg.norm(spatial_domain.step)
    dt = temporal_domain.step[0]
    
    def check_condition(condition: bool, message: str):
        if not condition:
            raise ValueError(f"Invalid configuration: {message}")
    
    if isinstance(initial_condition, GaussianWavePacket):
        lamb = 2 * np.pi / initial_condition.wave_number
        sigma = initial_condition.sigma
        omega = initial_condition.omega
        v_g = initial_condition.group_velocity
        
        # Spatial conditions
        check_condition(4 * ds < lamb, f"Spatial resolution too low: dx={ds}, λ={lamb}")
        check_condition(4 * sigma < L, f"Domain too small: σ={sigma}, L={L}")
        
        # Temporal conditions
        check_condition(4 * dt < 2*np.pi/omega, f"Time step too large: dt={dt}, T={2*np.pi/omega}")
        check_condition(4 * 2*np.pi/omega < L/v_g, f"Domain crossing time too short")
        
        # CFL condition
        check_condition(v_g * dt / ds <= 1, f"CFL condition violated: {v_g * dt / ds} > 1")
    
    if potential:
        if isinstance(potential, Wall):
            check_condition(4 * ds < potential.width.min(), f"Wall not resolved: dx={ds}, width={potential.width.min()}")
            # check_condition(2 * dt < 1/(2*potential.V_0), f"Time step too large for potential: dt={dt}, V_0={potential.V_0}")
            
    return True