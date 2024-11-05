"""
This module defines the `DipoleElectricFieldPotential` class, which represents the interaction 
between an electric dipole and a time-varying electric field, and visualizes the interaction 
potential in either 1D or 2D.

The interaction Hamiltonian is given by: \( H_{\text{int}} = -\vec{p} \cdot \vec{E}(t) \).

Notes
-----
This class extends the `TimeDependentPotential` base class and provides a concrete 
implementation of a time-dependent potential that varies sinusoidally in time.
"""

import numpy as np
import matplotlib.pyplot as plt

from schrodinger_engine.potentials import TimeDependentPotential
from schrodinger_engine.utils import Domain

class DipoleElectricFieldPotential(TimeDependentPotential):
    r"""
    Represent the interaction between an electric dipole and a time-dependent electric field.

    This potential is based on the following interaction Hamiltonian:
    .. math:: H_{\text{int}} = -\vec{p} \cdot \vec{E}(t)
    where :math:`\vec{p}` is the dipole moment vector and :math:`\vec{E}(t)` is the time-dependent electric field.
    
    Attributes
    ----------
    p : float
        The magnitude of the electric dipole moment.
    E_0 : float
        The peak amplitude of the electric field.
    f : float
        The frequency of oscillation of the electric field.
    theta : float
        Angle between the dipole moment and the electric field (in radians).
    dim : int
        Dimension of the potential, either 1 or 2.
    """

    def __init__(self, p, E_0, f, dim: int, theta=0.0) -> None:
        """
        Initialize the potential.

        Parameters
        ----------
        p : float
            The magnitude of the dipole moment.
        E_0 : float 
            The peak amplitude of the electric field.
        f : float
            The frequency of the electric field oscillations.
        dim : int
            Dimension of the potential (1 for 1D, 2 for 2D).
        theta : float, optional
            Angle between the electric dipole and the electric field in radians.
            Default is 0.0, which implies alignment between the field and the dipole.
        """
        self.p = p
        self.E_0 = E_0
        self.f = f
        self.theta = theta
        self.dim = self._validate_dim(dim)
        
    def __call__(self, x: float | np.ndarray, t: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the potential at specified spatial coordinate(s) `x` and time(s) `t`.

        Parameters
        ----------
        x : float or np.ndarray
            Spatial coordinate(s) where the potential is evaluated.
        t : float or np.ndarray
            Time or array of time values at which to evaluate the potential.

        Returns
        -------
        float or np.ndarray
            The evaluated potential at the specified coordinates and time(s).
        """
        x = np.asarray(x)
        t_is_scalar = np.isscalar(t)
        
        cos_theta = np.cos(self.theta)
        omega = 2 * np.pi * self.f
        E_term = -self.p * self.E_0 * cos_theta  # Constant term associated with the field strength and alignment
        
        if not t_is_scalar:
            t = np.asarray(t)
        
        match self.dim:
            case 1:
                if t_is_scalar:
                    return E_term * np.cos(omega * t) * np.ones_like(x)
                return E_term * np.cos(omega * t)[:, np.newaxis] * np.ones_like(x)
            case 2:
                if t_is_scalar:
                    return E_term * np.cos(omega * t) * np.ones_like(x[:, 0])
                return E_term * np.cos(omega * t)[:, np.newaxis, np.newaxis] * np.ones_like(x[0])

if __name__ == "__main__":
    # Example usage of the DipoleElectricFieldPotential class
    V = DipoleElectricFieldPotential(p=2, E_0=3, f=0.2, dim=2)
    spatial_domain = Domain(boundaries=[(-10, 10), (-10, 10)], step=0.1)
    time_domain = Domain(boundaries=[(0, 10)], step=0.05)
    
    # Display the potential
    V.display(spatial_domain=spatial_domain, temporal_domain=time_domain)