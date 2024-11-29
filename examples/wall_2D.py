"""Example: Gaussian wave packet scattering on a potential wall."""

import numpy as np
from pss import (
    PeriodicSchrodingerSimulation,
    GaussianWavePacket,
    Wall,
    Domain,
    SimulationReader
)
from pss.config import GaussianConfig, WallConfig

def run_wall_2D_simulation():
    # Domain setup
    L, dx = 100, 0.15  # Spatial parameters
    # T, dt = 30, 0.01  # Temporal parameters
    T, dt = 40, 0.02  # Temporal parameters
    dim = 2
    
    spatial_domain = Domain(boundaries=[[-L/2, L/2]]*dim, step=dx)
    temporal_domain = Domain(boundaries=[[0, T]], step=dt)
    
    # Wave packet setup
    x_0 = -10 * np.ones(dim)  # Initial position
    k_0 = 2 * np.ones(dim) / np.sqrt(dim)  # Wave vector
    sigma = 5  # Packet width
    
    wave_packet = GaussianWavePacket(GaussianConfig(
        x_0=x_0, k_0=k_0, sigma=sigma, dim=dim
    ))
    
    # Potential wall setup
    wall = Wall(WallConfig(
        x_0=np.abs(2*x_0),  # Wall position
        V_0=10,  # Wall height
        width=30*np.ones(dim),  # Wall width
        dim=dim
    ))
    
    # Run simulation
    sim = PeriodicSchrodingerSimulation(
        spatial_domain=spatial_domain,
        temporal_domain=temporal_domain,
        initial_condition=wave_packet,
        potential=wall,
        overwrite=True,
        name = "wall_2D"
    )
    sim.run()
    
    # Visualize results
    reader = SimulationReader("results/wall_2D.h5")
    # reader.animate(prop='probability', save_path='scattering.mp4')
    reader.animate_solution(
        # prop_to_display='probability_density',
        prop_to_display='real',
        # split_view = True
        save_name = 'wall_2D_full_reflexion_real_part'
        )
    
    reader.animate_solution(
        # prop_to_display='probability_density',
        # prop_to_display='real',
        # split_view = True
        save_name = 'wall_2D_full_reflexion_probability_density'
        )

if __name__ == '__main__':
    run_wall_2D_simulation()