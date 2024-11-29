"""Base potential classes for PSS simulations."""
from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pss.utils.domain import Domain

class Potential(ABC):
    """Abstract base class for static potentials."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate potential at given points."""
        pass
    
    def _validate_dim(self, dim: int) -> int:
        """Validate spatial dimension."""
        if dim not in [1, 2]:
            raise ValueError("Dimension must be 1 or 2")
        return dim
    
    def display(self, domain: Domain) -> None:
        """Display potential in real space."""
        mesh = domain.get_mesh()
        V = self(mesh)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if self.dim == 2 else None)
        
        if self.dim == 1:
            ax.plot(mesh, V)
            ax.set(xlabel=r"$x$", ylabel=r"$V(x)$")
        else:
            V = V.reshape(mesh[0].shape)
            ax.plot_surface(
                *mesh, V,
                cmap='cmr.lavender',
                alpha=0.85,
                linewidth=0,
                antialiased=True
            )
            ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$V(x,y)$")
        
        plt.show()

class TimeDependentPotential(Potential):
    """Abstract base class for time-dependent potentials."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        """Evaluate potential at given points and time."""
        pass