"""
This module defines a `Wall` potential class for simulating rectangular barrier potentials
in quantum mechanical systems, designed for 1D and 2D applications. 

The `Wall` class inherits from `Potential` and can be used to specify a wall potential of 
given height, width, and location in a defined domain. It supports evaluation of potential
energy at arbitrary points and can be visualized within a specified domain.

Usage Example
-------------
To create a 2D wall potential and display it over a specified domain:

>>> from schrodinger_engine.utils.domain import Domain
>>> wall_2d = Wall(x_0=[0, 0], V_0=10, width=(2, 4), dim=2)
>>> domain_2d = Domain(boundaries=[(-10, 10), (-10, 10)], N=[200, 200])
>>> wall_2d.display(domain=domain_2d)
>>> plt.show()
"""
from schrodinger_engine.potentials import Potential
from schrodinger_engine.utils.domain import Domain
import numpy as np
import matplotlib.pyplot as plt

class Wall(Potential):
    """
    Represents a rectangular wall potential, which can be used in one or two-dimensional 
    quantum mechanical simulations. This wall potential imposes a barrier of constant 
    potential energy within a specified region of space.
    
    Attributes
    ----------
    x_0 : np.ndarray
        Center of the wall converted to an array format.
    V_0 : float
        Height of the wall.
    dim : int
        Dimension of the domain, either 1D or 2D.
    width : np.ndarray
        Width(s) of the wall in each dimension.
    """
    def __init__(self, x_0: float|np.ndarray, V_0: float, width: float|tuple, dim: int):
        """
        Initialize a wall potential with specified center, height, width, and dimension.

        Parameters
        ----------
        x_0 : float or np.ndarray
            Center of the wall.
        V_0 : float
            Height of the wall.
        width : float or tuple of floats
            Width of the wall in each dimension.
        dim : int
            Dimension of the wall, either 1 for 1D or 2 for 2D.
        """
        self.x_0 = np.asarray(x_0)
        self.V_0 = V_0
        self.dim = self._validate_dim(dim)
        self.width = np.array(width) if isinstance(width, (list, tuple, np.ndarray)) else np.array([width] * self.dim)
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy of the wall at specified positions.

        Parameters
        ----------
        x : np.ndarray
            Array of positions where the potential energy is to be evaluated.
            For 1D, this is a 1D array of coordinates.
            For 2D, this is a 2D array of shape (2, N) representing N positions in 2D space.

        Returns
        -------
        np.ndarray
            An array of potential energy values, with each element corresponding to
            a position in the input array. Inside the wall, the value is `V_0`; outside,
            it is zero.
        """
        match self.dim:
            case 1:
                r = np.abs(x - self.x_0)
                inside_wall = (r <= self.width / 2)
            case 2:
                X = (x.T - self.x_0).T
                inside_wall = np.logical_and(np.abs(X[0]) <= self.width[0]/2, np.abs(X[1]) <= self.width[1]/2)
        potential_energy = np.where(inside_wall, self.V_0, 0.0).flatten()
        return potential_energy



if __name__ == '__main__':
    wall_1d = Wall(x_0=0, V_0=10, width=1, dim=1)
    domain_1d = Domain(boundaries=[(-10, 10)], step=0.1)
    wall_1d.display(domain=domain_1d)
    plt.show()

    domain_2d = Domain(boundaries=[(-10, 10), (-10, 10)], N=[200, 200])
    wall_2d = Wall(x_0=[0, 0], V_0=10, width=(2, 4), dim=2)
    wall_2d.display(domain=domain_2d)
    plt.show()
    