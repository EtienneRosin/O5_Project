from schrodinger_engine.potentials import Potential, Slit
from schrodinger_engine.utils import Domain

import numpy as np
import matplotlib.pyplot as plt

class DoubleSlit(Potential):
    """ Represent a double slit in the x direction. It is 2D potential.
    """
    def __init__(
        self,
        x_0: np.ndarray,
        width: float,
        depth: float,
        a: float,
        V_0: float = 1e1) -> None:
        
        # x_0: float|np.ndarray, V_0: float, width: float|tuple, dim: int
        self.x_0 = np.asarray(x_0)
        self.a = a
        self.width = width
        self.depth = depth
        self.dim = 2
        self.V_0 = V_0
        
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy of the double slit at specified positions.

        Parameters
        ----------
        x : np.ndarray
            Array of positions where the potential energy is to be evaluated.

        Returns
        -------
        np.ndarray
            An array of potential energy values, with each element corresponding to
            a position in the input array. Inside the double slit walls, the value is `V_0`; outside,
            it is zero.
        """
        x = np.asarray(x)
        X = (x.T - self.x_0).T
        inside_wall = np.logical_and(
            np.logical_and(np.abs(X[0]) <= self.depth / 2, np.abs(X[1] - self.a/2) >= self.width / 2), 
            np.logical_and(np.abs(X[0]) <= self.depth / 2, np.abs(X[1] + self.a/2) >= self.width / 2))
        potential_energy = np.where(inside_wall, self.V_0, 0.0).flatten()
        return potential_energy
    

if __name__ == '__main__':
    slit_position = np.array([0.0, 0.0])  # Center of the slit at origin
    slit_width = 0.1  # Width of the slit
    slit_depth = 0.5  # Depth of the potential
    a = 0.5

    double_slit = DoubleSlit(x_0=slit_position, width=slit_width, depth=slit_depth, a = a)
    domain_2d = Domain(boundaries=[(-2.5, 2.5), (-2.5, 2.5)], step=0.005)
    double_slit.display(domain_2d)
    plt.show()