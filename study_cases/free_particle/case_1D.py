from schrodinger_engine.simulation import PeriodicSchrodingerSimulation
from schrodinger_engine.initial_conditions import GaussianWavePacket
from schrodinger_engine.visualization_tools import SimulationReader
from schrodinger_engine.utils import Domain

import numpy as np


# Parameters ------------------------------------------------------------------
simulation_name = "free_particle_1D"

L = 100
dx = 0.1

T = 30
dt = 0.01
dim = 1

boundaries = [[-L/2, L/2] for _ in range(dim)]

x_0 = -10*np.ones(dim)
wave_number = 2
k_0 = np.ones(dim)
k_0 = wave_number * k_0 / np.linalg.norm(k_0)
a = 30 * np.ones(dim)
V_0 = 1e1

sigma = 5
spatial_domain = Domain(boundaries=boundaries, step=dx)
temporal_domain = Domain(boundaries=[[0, T]], step=dt)
gauss_wave_packet = GaussianWavePacket(x_0=x_0, k_0=k_0, sigma=sigma, dim=dim)


sim = PeriodicSchrodingerSimulation(
    spatial_domain=spatial_domain, 
    temporal_domain=temporal_domain, 
    initial_condition=gauss_wave_packet,
    name = simulation_name,
    # potential=Wall(x_0 = np.abs(2*x_0), V_0=V_0, width=a, dim=dim),
    overwrite=True
)
sim.run()


prop_to_display = 'real'
animation_name = f"{sim._save_folder}/{simulation_name}_{prop_to_display}"
reader = SimulationReader(file_path = sim._save_path)
reader.animate_solution(
    prop_to_display='real',
    # save_name = animation_name
    )

