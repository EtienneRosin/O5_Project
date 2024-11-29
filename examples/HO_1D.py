"""Example: Gaussian wave packet scattering on a potential wall."""

import numpy as np
from pss import (
    PeriodicSchrodingerSimulation,
    GaussianWavePacket,
    Wall,
    HarmonicOscillator,
    Domain,
    SimulationReader
)
from pss.config import GaussianConfig, HarmonicConfig, WallConfig

def run_HO_1D_simulation():
    # Domain setup
    L, dx = 1000, 0.15  # Spatial parameters
    # T, dt = 30, 0.01  # Temporal parameters
    T, dt = 40, 0.02  # Temporal parameters
    dim = 1
    
    spatial_domain = Domain(boundaries=[[-L/2, L/2]]*dim, step=dx)
    temporal_domain = Domain(boundaries=[[0, T]], step=dt)
    
    # Wave packet setup
    x_0 = np.zeros(dim)  # Initial position
    direction = np.ones(dim)
    wavenumber = 7
    k_0 = wavenumber * direction / np.linalg.norm(direction)  # Wave vector
    sigma = 2  # Packet width
    
    wave_packet = GaussianWavePacket(GaussianConfig(
        x_0=x_0, k_0=k_0, sigma=sigma, dim=dim
    ))
    
    # Potential wall setup
    ho = HarmonicOscillator(
        HarmonicConfig(x_0=25*np.ones(dim), V_0=1, dim=1, omega=1e10))
    
    
    wall = Wall(WallConfig(
        x_0=50*np.ones(dim),  # Wall position
        V_0=20,  # Wall height
        width=30*np.ones(dim),  # Wall width
        dim=dim
    ))
    
    # Run simulation
    sim = PeriodicSchrodingerSimulation(
        spatial_domain=spatial_domain,
        temporal_domain=temporal_domain,
        initial_condition=wave_packet,
        potential=ho,
        overwrite=True,
        name = "ho_1D"
    )
    sim.run()
    
    # Visualize results
    reader = SimulationReader("results/ho_1D.h5")
    # reader.animate(prop='probability', save_path='scattering.mp4')
    reader.animate_solution(
        # prop_to_display='probability_density',
        prop_to_display='real',
        # split_view = True
        # save_name = 'ho_1D_real_part'
        )
    
    reader.animate_solution(
        # prop_to_display='probability_density',
        # prop_to_display='real',
        # split_view = True
        # save_name = 'ho_1D_probability_density'
        )

if __name__ == '__main__':
    run_HO_1D_simulation()