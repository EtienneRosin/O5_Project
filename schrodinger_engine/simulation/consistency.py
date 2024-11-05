# -*- coding: utf-8 -*-
from schrodinger_engine.utils import Domain, FourierBasis
from schrodinger_engine.initial_conditions import InitialCondition, PlaneWave, GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential, Wall
import numpy as np

def configuration_is_consistent(
    spatial_domain: Domain, 
    temporal_domain: Domain, 
    initial_condition: InitialCondition, 
    potential: Potential = None,
    spatial_factor = 4,
    temporal_factor = 4
    ) -> bool:
    
    L = spatial_domain.length.min()
    ds = np.linalg.norm(spatial_domain.step)
    T = temporal_domain.length[0]
    dt = temporal_domain.step[0]
    
    # Free particle consistency case --------------------------------
    match initial_condition.__class__.__name__:
            case 'GaussianWavePacket':
                lamb = 2 * np.pi / initial_condition.wave_number # spatial wave length
                sigma = initial_condition.sigma # width
                omega = initial_condition.omega
                v_g = initial_condition.group_velocity
                
                # characteristic lengths ------------------
                if not (spatial_factor * ds < lamb):
                    raise ValueError(f"Non-satisfied condition : Δx << λ  (here Δx = {ds:.3g} and λ = {lamb:.3g}).")
                
                # if not (lamb < sigma):
                #     raise ValueError(f"Non-satisfied condition : λ < σ  (here λ = {lamb:.3g} and σ = {sigma:.3g}).")
                
                if not (spatial_factor * sigma < L):
                    raise ValueError(f"Non-satisfied condition : σ << L  (here σ = {sigma:.3g} and L = {L:.3g}).")
                
                # characteristic times --------------------
                if not (temporal_factor * dt < 2 * np.pi / omega):
                    raise ValueError(f"Non-satisfied condition : Δt << 2π/ω  (here Δt = {lamb:.3g} and 2π/ω = {2 * np.pi / omega:.3g}).")
                
                if not (temporal_factor * (2 * np.pi / omega) < L / v_g):
                    raise ValueError(f"Non-satisfied condition : 2π/ω << L/v_g (here 2π/ω = {2 * np.pi / omega:.3g} and L/v_g = {L/v_g:.3g}).")
                
                # if not (temporal_factor * L / v_g < T):
                #     raise ValueError(f"Non-satisfied condition : L/v_g << T (here L/v_g = {L/v_g:.3g} and T = {T:.3g}).")
            case _:
                raise ValueError(f"Initial condition '{initial_condition.__class__.__name__}' has no consistency test.")

    if potential:
        match potential.__class__.__name__:
            case 'Wall':
                if not (spatial_factor * ds < potential.width.min()):
                        raise ValueError(f"Non-satisfied condition : ds < a  (here λ = {ds:.3g} and a = {potential.width.min():.3g}).")
            
                if not (temporal_factor * dt < 1 / potential.V_0):
                    raise ValueError(f"Non-satisfied condition : dt < ℏ/V_0  (here dt = {dt:.3g} and ℏ/V_0 = {1/potential.V_0:.3g}).")
            case 'Slit':
                print(f"Slit has no consistency test")
                pass
            case 'DoubleSlit':
                print(f"DoubleSlit has no consistency test")
            case _:
                raise ValueError(f"potential '{potential.__class__.__name__}' has no consistency test.")

    return True


if __name__ == "__main__":
    T = 10
    N_t = 400
    L = 50
    N_x = 1000
    
    # # Omega = SpatialDomain(boundaries = [-L/2, L/2], N = N_x)
    # # temporal = TemporalDomain(t_end = T, N = N_t)
    
    # # psi_0 = PlaneWave(k = 1)
    # # GaussianWavePacket(x_0=0.0, sigma=2.0, k=5.0)
    
    # print(configuration_is_consistent(
    #     spatial_domain = Domain(boundaries = [[-L/2, L/2]], N = N_x),
    #     temporal_domain = Domain(boundaries = [[0, T]], N = N_t),
    #     initial_condition = GaussianWavePacket(x_0=0.0, k_0=10, sigma=1.0, dim=1)
    #     )
    #       )