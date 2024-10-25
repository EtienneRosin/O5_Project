# -*- coding: utf-8 -*-
from schrodinger_engine.domains import SpatialDomain, TemporalDomain
from schrodinger_engine.fourier import FourierBasis
from schrodinger_engine.initial_conditions import InitialCondition, PlaneWave, GaussianWavePacket
from schrodinger_engine.potentials import Potential, TimeDependentPotential, Wall
import numpy as np

def configuration_is_consistent(
    spatial_domain: SpatialDomain, 
    temporal_domain: TemporalDomain, 
    initial_condition: InitialCondition, 
    potential: Potential = None,
    spatial_factor = 4,
    temporal_factor = 4
    ) -> bool:
    L = spatial_domain.width
    dx = spatial_domain.step
    T = temporal_domain.width
    dt = temporal_domain.step
    
    # factor = 4 

    # Longueurs caractéristiques
    if potential is None:
        match initial_condition.__class__.__name__:
            case 'PlaneWave':
                lamb = initial_condition.lamb   # wave length (in space)
                k = 2 * np.pi / lamb            # wave number (in space)
                omega = 0.5*k**2        # angular frequency (in time)
                
                if not (spatial_factor * dx < lamb):
                    raise ValueError(f"Non-satisfied condition : Δx << λ  (here Δx = {dx:.3g} and λ = {lamb:.3g}).")
                
                if not (spatial_factor * lamb < L):
                    raise ValueError(f"Non-satisfied condition : λ << L  (here λ = {lamb:.3g} and L = {L:.3g}).")
                
                if not (temporal_factor * dt < 2 * np.pi / omega): # time step - temporal period of the wave
                    raise ValueError(f"Non-satisfied condition : Δt << 2π/ω  (here Δt = {lamb:.3g} and 2π/ω = {2 * np.pi / omega:.3g}).")
                
                if not (temporal_factor * (2 * np.pi / omega) < 2 * L / k): # temporal period of the wave - period of propagation in the domain 
                    raise ValueError(f"Non-satisfied condition : 2π/ω << 2L/k  (here 2π/ω = {2 * np.pi / omega:.3g} and 2L/k = {2 * L / k:.3g}).")


            case 'GaussianWavePacket':
                lamb = initial_condition.lamb   # wave length
                k = 2 * np.pi / lamb            # wave number
                sigma = initial_condition.sigma # gaussian width
                omega = 0.5*k**2        # angular frequency (in time)

                if not (spatial_factor * dx < lamb):
                    raise ValueError(f"Non-satisfied condition : Δx << λ  (here Δx = {dx:.3g} and λ = {lamb:.3g}).")
    
                # if not (lamb < sigma):
                #     raise ValueError(f"Non-satisfied condition : λ < σ  (here λ = {lamb:.3g} and σ = {sigma:.3g}).")
                
                if not (spatial_factor * lamb < L):
                    raise ValueError(f"Non-satisfied condition : λ << L  (here λ = {lamb:.3g} and L = {L:.3g}).")
                
                if not (spatial_factor * sigma < L):
                    raise ValueError(f"Non-satisfied condition : σ << L  (here σ = {sigma:.3g} and L = {L:.3g}).")
                
                if not (temporal_factor * dt < 2 * np.pi / omega): # time step - temporal period of the wave
                    raise ValueError(f"Non-satisfied condition : Δt << 2π/ω  (here Δt = {lamb:.3g} and 2π/ω = {2 * np.pi / omega:.3g}).")
                
                if not (temporal_factor * (2 * np.pi / omega) < 2 * L / k): # temporal period of the wave - period of propagation in the domain 
                    raise ValueError(f"Non-satisfied condition : 2π/ω << 2L/k  (here 2π/ω = {2 * np.pi / omega:.3g} and 2L/k = {2 * L / k:.3g}).")

            case _:
                raise ValueError(f"Initial condition '{initial_condition.__class__.__name__}' has no consistency test.")
    else:
        # # Ajouter une vérification pour les potentiels ici si nécessaire
        # # Exemples: vérifier la largeur ou l'amplitude du potentiel
        # if hasattr(potential, 'characteristic_length'):
        #     V_char_length = potential.characteristic_length()  # Longueur caractéristique du potentiel
        #     if not (dx < V_char_length / factor < L):
        #         raise ValueError(f"Conditions non satisfaites pour le potentiel: {dx} < {V_char_length / factor} < {L}.")
        pass

    # Vérification des échelles de temps
    # tau = initial_condition.get_characteristic_time()  # Méthode hypothétique pour récupérer l'échelle de temps caractéristique
    # if not (dt < tau / factor < T):
    #     raise ValueError(f"Conditions non satisfaites pour le temps: {dt} < {tau / factor} < {T}.")

    return True


if __name__ == "__main__":
    T = 10
    N_t = 200
    L = 50
    N_x = 1000
    
    # Omega = SpatialDomain(boundaries = [-L/2, L/2], N = N_x)
    # temporal = TemporalDomain(t_end = T, N = N_t)
    
    # psi_0 = PlaneWave(k = 1)
    # GaussianWavePacket(x_0=0.0, sigma=2.0, k=5.0)
    
    print(configuration_is_consistent(
        spatial_domain = SpatialDomain(boundaries = [-L/2, L/2], N = N_x),
        temporal_domain = TemporalDomain(t_end = T, N = N_t),
        initial_condition = GaussianWavePacket(x_0=0.0, sigma = 2.0, lamb = 1.9)
        )
          )