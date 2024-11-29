"""
PSS: Periodic Schrödinger Solver
--------------------------------
A quantum mechanics simulation package focused on solving the time-dependent 
Schrödinger equation with periodic boundary conditions.
"""

__version__ = "0.1.0"

# Core functionality
from .core import PeriodicSchrodingerSimulation

# Initial conditions
from .waves.gaussian import GaussianWavePacket
# from .waves.plane import PlaneWave

# Potentials
from .potentials import Wall, Slit, HarmonicOscillator

# Utilities
from .utils import Domain, FourierBasis, SimulationReader


__all__ = [
    'PeriodicSchrodingerSimulation',
    'GaussianWavePacket',
    # 'PlaneWave',
    'Wall',
    'Slit', 
    'HarmonicOscillator',
    'Domain',
    'FourierBasis',
    'SimulationReader'
]