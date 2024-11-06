from schrodinger_engine.simulation import PeriodicSchrodingerSimulation
from schrodinger_engine.initial_conditions import GaussianWavePacket
from schrodinger_engine.potentials import Slit
from schrodinger_engine.visualization_tools import SimulationReader
from schrodinger_engine.utils import Domain

import numpy as np


# Parameters ------------------------------------------------------------------
simulation_name = "single_slit"

L = 25

dx = 0.01
# dx = 0.05

T = 5
dt = 0.005
dim = 2

slit_position = np.array([0.0, 0.0])  # Center of the slit at origin
slit_width = 0.05  # Width of the slit
slit_depth = 1  # Depth of the potentia


sigma = 3 * slit_width

boundaries = [[-2*L, 2*L], [-L/2, L/2]]

x_0 = [-L/4, 0]
# wave_number = 1.5
# wave_number = 1
wave_number = 10
k_0 = np.array([1, 0])
k_0 = wave_number * k_0 / np.linalg.norm(k_0)
# V_0 = 1e0
V_0 = 1e10


spatial_domain = Domain(boundaries=boundaries, step=dx)
temporal_domain = Domain(boundaries=[[0, T]], step=dt)
gauss_wave_packet = GaussianWavePacket(x_0=x_0, k_0=k_0, sigma=sigma, dim=dim)


# sim = PeriodicSchrodingerSimulation(
#     spatial_domain=spatial_domain, 
#     temporal_domain=temporal_domain, 
#     initial_condition=gauss_wave_packet,
#     name = simulation_name,
#     potential=Slit(x_0=slit_position, width=slit_width, depth=slit_depth, V_0=V_0),
#     overwrite=True,
#     consistency_check = False
# )
# sim.run()


prop_to_display = 'probability_density'
# prop_to_display = 'real'
animation_name = f"{'results'}/{simulation_name}_{prop_to_display}"
reader = SimulationReader(file_path = f"{'results'}/{simulation_name}.h5")
reader.animate_solution(
    prop_to_display=prop_to_display,
    # save_name = animation_name
    )

