"""
This submodule `potentials` contains classes that define different potential energy models 
for use in quantum mechanics simulations. 

Each potential class encapsulates the mathematical formulation of a specific type of potential 
energy function, which can be static or time-dependent, and is designed to interact seamlessly 
with the `schrodinger_engine` simulation framework.

Classes
-------
Potential : ABC
    Abstract base class for defining static potential energy functions in 1D or 2D domains.
    
TimeDependentPotential : ABC
    Abstract base class for defining time-dependent potential energy functions in 1D or 2D domains.
    
HarmonicOscillator : Potential
    Implements a harmonic oscillator potential, modeling quadratic confinement around an 
    equilibrium position.

Wall : Potential
    Models an infinite potential wall to confine particles within a specified region.
    
DipoleElectricFieldPotential : TimeDependentPotential
    Represents the interaction potential of an electric dipole in an oscillating electric field.
    
Slit : Potential
    Represents a single-slit barrier, often used to simulate wave interference and diffraction.
    
DoubleSlit : Potential
    Models a double-slit barrier, widely used in quantum mechanics experiments to demonstrate 
    wave-particle duality.

Notes
-----
The classes in this module are designed to be used as components in quantum mechanics simulations. 
They provide specific potential functions that can be visualized, evaluated, and integrated into 
Schrödinger equation solvers or other simulation tools within the `schrodinger_engine` framework.

Examples
--------
To create and visualize a potential, instantiate the desired potential class and pass a 
`Domain` instance to the `display` method. For example:

>>> import matplotlib.pyplot as plt
>>> from schrodinger_engine.utils import Domain
>>> from schrodinger_engine.potentials import HarmonicOscillator
>>> domain = Domain(boundaries=[(-10, 10)], step=0.1)
>>> harmonic_oscillator = HarmonicOscillator(x_0=0, omega=1, dim=1)
>>> harmonic_oscillator.display(domain)
>>> plt.show()
"""

from .potential import Potential, TimeDependentPotential
from .harmonic_oscillator import HarmonicOscillator
from .wall import Wall
from .electric_field import DipoleElectricFieldPotential
from .slit import Slit
from .double_slit import DoubleSlit