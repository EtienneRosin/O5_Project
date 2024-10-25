from numpy import ndarray
from schrodinger_engine.potentials import Potential, TimeDependentPotential

import numpy as np
import matplotlib.pyplot as plt

class DipoleElectricFieldPotential(TimeDependentPotential):
    """
    Represent the interaction between an electric dipole and an electric field.
    
    The interaction hamiltonian is given by : H_int = - \vec{p}\dot\vec{E}(t).
    """
    def __init__(self, p, E_0, f, theta = 0.0) -> None:
        """
        Initialize the potential.
        
        Parameters
        ----------
        p : float
            (norm of the) dipole moment.
        E_0 : float 
            Magnitude of the electric field.
        f : float
            Frequency of the electric field.
        theta : float
            Angle between the electric dipole and the electric field (default : they are aligned).
        """
        self.p = p
        self.E_0 = E_0
        self.f = f
        self.theta = theta
        
    def __call__(self, x: float | ndarray, t: float | ndarray) -> float | ndarray:
        cos_theta = np.cos(self.theta)
        x = np.asarray(x)
        omega = 2 * np.pi * self.f
        # If t is a scalar, return a 1D array for x
        if np.isscalar(t):
            return - self.p * self.E_0 * cos_theta * np.cos(omega * t) * np.ones_like(x)

        # If t is an array, return a 2D array for (t, x)
        t = np.asarray(t)
        return - self.p * self.E_0 * cos_theta * np.cos(omega * t)[:, np.newaxis] * np.ones_like(x)


if __name__ == "__main__":
    T = 10
    lst_t = np.linspace(start=0, stop=T, num=100, endpoint=True)
    lst_x = np.linspace(-15, 15, 200)
    
    V = DipoleElectricFieldPotential(p=2, E_0=3, f=0.2)
    
    V.display()
