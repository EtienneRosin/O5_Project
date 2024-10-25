from schrodinger_engine.potentials import Potential
from schrodinger_engine.domains.test_domain import Domain
import numpy as np
import matplotlib.pyplot as plt

class HarmonicOscillator(Potential):
    """
    Represents an harmonic oscillator
    """
    def __init__(self, x_0: np.ndarray, omega: float, dim: int):
        """
        Initialize the harmonic oscillator.

        Parameters
        ----------
        x_0: np.ndarray or float
            initial equilibrium position (1D) or positions (2D).
        omega: float
            angular frequency.
        dim: int
            Dimension of the domain.
        """
        self.x_0 = np.asarray(x_0)  # Convert x_0 to a numpy array
        self.omega = omega
        self.dim = self._validate_dim(dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy of the harmonic oscillator.

        Parameters
        ----------
        x: np.ndarray
            positions (1D or 2D).

        Returns
        -------
        Potential energy: np.ndarray
        """
        x = np.asarray(x)  # Ensure x is a numpy array
        if self.dim == 1:  # 1D case
            return 0.5 * (self.omega ** 2) * (x - self.x_0) ** 2
        elif self.dim == 2:  # 2D case
            if x.ndim == 3:
                x = np.stack((x[0], x[1]), axis=-1)
                x = x.reshape(-1, 2)
                # print(f"{inside_wall.shape = }")
                # print(f"{x.shape = }")
            return 0.5 * (self.omega ** 2) * np.sum((x - self.x_0) ** 2, axis=-1)
        else:
            raise ValueError("Input must be 1D or 2D array.")


if __name__ == '__main__':
    
    domain_1d = Domain(boundaries=[(-10, 10)], step=0.1)
    ho_1d = HarmonicOscillator(x_0=1, omega=2, dim=1)
    ho_1d.display(domain=domain_1d)
    plt.show()

    domain_2d = Domain(boundaries=[(-10, 10), (-10, 10)], N=[100, 100])    
    ho_2d = HarmonicOscillator(x_0=[-5, 1], omega=2, dim=2)
    print(f"{ho_2d(domain_2d.get_mesh()).shape = }")
    ho_2d.display(domain=domain_2d)
    plt.show()
    