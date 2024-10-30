from schrodinger_engine.potentials import Potential
from schrodinger_engine.utils.domain import Domain
import numpy as np
import matplotlib.pyplot as plt

class Wall(Potential):
    """
    Represents a Wall
    """
    def __init__(self, x_0: float|np.ndarray, V_0: float, width: float|tuple, dim: int):
        """
        @brief Initialize a wall potential
        @param x_0: center of the wall
        @param V_0: height of the wall
        @param b: width of the wall
        """
        """
        Initialize a wall potential.

        Parameters
        ----------
        x_0: np.ndarray or float
            initial center of the wall.
        V_0: float
            Height of the wall.
        width: tuple or float
            Width(s) of the wall.
        dim: int
            Dimension of the domain.
        """
        self.x_0 = np.asarray(x_0)  # Convert x_0 to a numpy array
        self.V_0 = V_0
        self.dim = self._validate_dim(dim)
        self.width = np.array(width) if isinstance(width, (list, tuple, np.ndarray)) else np.array([width] * self.dim)
        

    # def __call__(self, x: np.ndarray) -> np.ndarray:
    #     """
    #     Calculate the potential energy of the wall.
    #     Parameters
    #     ----------
    #     x: np.ndarray
    #         Positions (1D or 2D).
    #     Returns
    #     -------
    #     Potential energy: np.ndarray
    #     """
    #     x = np.asarray(x)  # Ensure x is a numpy array

    #     # if self.dim == 1:
    #     #     inside_wall = (x >= self.x_0 - self.width / 2) & (x <= self.x_0 + self.width / 2)
    #     # elif self.dim == 2:
    #     #     inside_wall = np.ones(x.shape[1:], dtype=bool)
    #     #     for i in range(self.dim):
    #     #         lower_bound = self.x_0[i] - self.width[i] / 2
    #     #         upper_bound = self.x_0[i] + self.width[i] / 2
    #     #         inside_wall &= (x[i, :, :] >= lower_bound) & (x[i, :, :] <= upper_bound)

    #     # potential_energy = np.where(inside_wall, self.V_0, 0.0)
    #     # return potential_energy
    #     inside_wall = np.ones(x[0].shape, dtype=bool)
    #     for i in range(self.dim):
    #         lower_bound = self.x_0[i] - self.width[i] / 2
    #         upper_bound = self.x_0[i] + self.width[i] / 2
    #         inside_wall &= (x[i] >= lower_bound) & (x[i] <= upper_bound)

    #     potential_energy = np.where(inside_wall, self.V_0, 0.0)
    #     return potential_energy
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy of the wall.
        
        Parameters
        ----------
        x: np.ndarray
            Positions (2D).
        
        Returns
        -------
        Potential energy: np.ndarray
        """
        match self.dim:
            case 1:
                r = np.abs(x - self.x_0)
                inside_wall = (r <= self.width / 2)
            case 2:
                X = (x.T - self.x_0).T  # Centrage par rapport à x_0
                inside_wall = np.logical_and(np.abs(X[0]) <= self.width[0]/2, np.abs(X[1]) <= self.width[1]/2)
                # np.stack((x[0], x[1]), axis=-1)
                # inside_wall = np.stack((x[0], x[1]), axis=-1)
                # print(f"{inside_wall.shape = }")
                # print(f"{x.shape = }")
        potential_energy = np.where(inside_wall, self.V_0, 0.0).flatten()
        return potential_energy



if __name__ == '__main__':
    # wall_1d = Wall(x_0=0, V_0=10, width=1, dim=1)
    # domain_1d = Domain(boundaries=[(-10, 10)], step=0.1)
    # wall_1d.display(domain=domain_1d)
    # plt.show()

    domain_2d = Domain(boundaries=[(-10, 10), (-10, 10)], N=[200, 200])
    wall_2d = Wall(x_0=[0, 0], V_0=10, width=(2, 4), dim=2)
    wall_2d.display(domain=domain_2d)
    plt.show()
    