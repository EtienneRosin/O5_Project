from schrodinger_engine.simulation import PeriodicSchrodingerSimulation
from schrodinger_engine.initial_conditions import GaussianWavePacket
from schrodinger_engine.potentials import Wall
from schrodinger_engine.visualization_tools import SimulationReader
from schrodinger_engine.utils import Domain

import numpy as np


# Parameters ------------------------------------------------------------------
simulation_name = "wall_1D"

L = 100
dx = 0.1

T = 50
dt = 0.01
dim = 2

boundaries = [[-L/2, L/2] for _ in range(dim)]

x_0 = -10*np.ones(dim)
# wave_number = 1.5
wave_number = 5
k_0 = np.ones(dim)
k_0 = wave_number * k_0 / np.linalg.norm(k_0)
a = 15 * np.ones(dim)
# V_0 = 1e0
V_0 = 1e1

sigma = 5
spatial_domain = Domain(boundaries=boundaries, step=dx)
temporal_domain = Domain(boundaries=[[0, T]], step=dt)
gauss_wave_packet = GaussianWavePacket(x_0=x_0, k_0=k_0, sigma=sigma, dim=dim)


# sim = PeriodicSchrodingerSimulation(
#     spatial_domain=spatial_domain, 
#     temporal_domain=temporal_domain, 
#     initial_condition=gauss_wave_packet,
#     name = simulation_name,
#     potential=Wall(x_0 = np.abs(2*x_0), V_0=V_0, width=a, dim=dim),
#     overwrite=True
# )
# sim.run()


prop_to_display = 'probability_density'
prop_to_display = 'real'
animation_name = f"{'results'}/{simulation_name}_{prop_to_display}"
reader = SimulationReader(file_path = f"{'results'}/{simulation_name}.h5")
reader.animate_solution(
    prop_to_display=prop_to_display,
    # save_name = animation_name
    )

