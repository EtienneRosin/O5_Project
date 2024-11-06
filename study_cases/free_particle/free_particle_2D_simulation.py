from schrodinger_engine.simulation import PeriodicSchrodingerSimulation
from schrodinger_engine.initial_conditions import GaussianWavePacket
from schrodinger_engine.visualization_tools import SimulationReader
from schrodinger_engine.utils import Domain

import numpy as np


# Parameters ------------------------------------------------------------------
simulation_name = "free_particle_2D"


L = 25
dx = 0.05
dx = 0.075
# dx = 0.025

T = 30
T = 15
dt = 0.01
dim = 2

boundaries = [[-L/2, L/2] for _ in range(dim)]

x_0 = -2.5*np.ones(dim)
wave_number = 5
wave_number = 2.5
k_0 = np.array([1, 0.25])
k_0 = wave_number * k_0 / np.linalg.norm(k_0)

sigma = 2
spatial_domain = Domain(boundaries=boundaries, step=dx)
temporal_domain = Domain(boundaries=[[0, T]], step=dt)
gauss_wave_packet = GaussianWavePacket(x_0=x_0, k_0=k_0, sigma=sigma, dim=dim)


sim = PeriodicSchrodingerSimulation(
    spatial_domain=spatial_domain, 
    temporal_domain=temporal_domain, 
    initial_condition=gauss_wave_packet,
    name = simulation_name,
    overwrite=True
)
sim.run()


prop_to_display = 'real'
prop_to_display = 'probability_density'
animation_name = f"{'results'}/{simulation_name}_{prop_to_display}"
reader = SimulationReader(file_path = f"{'results'}/{simulation_name}.h5")
reader.animate_solution(
    prop_to_display=prop_to_display,
    # save_name = animation_name
    )

