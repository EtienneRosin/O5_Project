"""
This module defines the `HarmonicOscillator` class, a specific implementation of the 
`Potential` abstract base class that represents a harmonic oscillator in either 1D or 2D space. 

The harmonic oscillator potential is widely used in quantum mechanics, representing 
a particle oscillating around an equilibrium position with a quadratic potential energy 
function.

Notes
-----
This class is derived from `Potential`, an abstract base class for defining static potentials, 
and provides a specific potential energy calculation and visualization for harmonic oscillators.
"""

from schrodinger_engine.potentials import Potential
from schrodinger_engine.utils.domain import Domain
import numpy as np
import matplotlib.pyplot as plt

class HarmonicOscillator(Potential):
    """
    Harmonic Oscillator Potential.

    This class models a harmonic oscillator, where the potential energy is a quadratic 
    function of the distance from an equilibrium position. 

    Attributes
    ----------
    x_0 : np.ndarray
        The equilibrium position of the oscillator.
    omega : float
        The angular frequency of the oscillator.
    dim : int
        The spatial dimensionality of the oscillator, either 1 or 2.
    """

    def __init__(self, x_0: np.ndarray, omega: float, dim: int):
        """
        Initialize the harmonic oscillator with its equilibrium position, frequency, and dimensionality.

        Parameters
        ----------
        x_0 : np.ndarray or float
            The initial equilibrium position in 1D or an array of positions in 2D.
        omega : float
            The angular frequency of the oscillator, determining the "stiffness" of the potential.
        dim : int
            The spatial dimension of the oscillator (1 or 2).
        """
        self.x_0 = np.asarray(x_0)  # Ensure `x_0` is a numpy array for consistent calculations.
        self.omega = omega
        self.dim = self._validate_dim(dim)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate the potential energy of the harmonic oscillator at given positions.

        Parameters
        ----------
        x : np.ndarray
            Array of positions where the potential energy is evaluated. This can be a 
            1D array for 1D positions or a 2D array for 2D positions.

        Returns
        -------
        np.ndarray
            The potential energy values at the input positions.

        Notes
        -----
        The potential energy function \( V(x) \) for the harmonic oscillator is given by:
        - In 1D: :math:`V(x) = \frac{1}{2} \omega^2 (x - x_0)^2` \)`
        - In 2D: :math:`V(x, y) = \frac{1}{2} \omega^2 [(x - x_0)^2 + (y - y_0)^2]`
        """
        x = np.asarray(x)  # Ensure `x` is a numpy array
        if self.dim == 1:
            return 0.5 * (self.omega ** 2) * (x - self.x_0) ** 2
        elif self.dim == 2:
            if x.ndim == 3:
                x = np.stack((x[0], x[1]), axis=-1)
                x = x.reshape(-1, 2)
            return 0.5 * (self.omega ** 2) * np.sum((x - self.x_0) ** 2, axis=-1)
        else:
            raise ValueError("Input must be 1D or 2D array.")

if __name__ == '__main__':
    # Example usage of HarmonicOscillator in 1D
    domain_1d = Domain(boundaries=[(-10, 10)], step=0.1)
    ho_1d = HarmonicOscillator(x_0=1, omega=2, dim=1)
    ho_1d.display(domain=domain_1d)
    plt.show()

    # Example usage of HarmonicOscillator in 2D
    domain_2d = Domain(boundaries=[(-1, 1), (-1, 1)], N=[100, 100])    
    ho_2d = HarmonicOscillator(x_0=[0, 0], omega=2, dim=2)
    ho_2d.display(domain=domain_2d)
    plt.show()