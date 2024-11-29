"""Base wave packet classes for PSS simulations."""

from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from pss.utils import Domain
# from ..utils.viz.plotting import plot_1D_complex_field, plot_2D_complex_field

class InitialCondition(ABC):
    """Abstract base class for quantum wave packets."""
    
    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Evaluate wave function at given points."""
        pass
    
    def _standardize_input(self, x: np.ndarray) -> np.ndarray:
        """Standardize input array to (dim, N) shape."""
        x = np.atleast_2d(x)
        if x.shape[0] != self.dim:
            x = x.T
        if x.ndim == 3:  # Handle meshgrid input
            x = np.vstack([x[i].ravel() for i in range(self.dim)])
        return x
    
    def display(self, domain: Domain = None) -> None:
        """Display wave function in real space."""
        if domain is None:
            domain = Domain(
                boundaries=[[-40, 40]]*self.dim,
                N=[1000]*self.dim
            )
        
        mesh = domain.get_mesh()
        psi = self(mesh)
        
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if self.dim == 2 else None)
        
        if self.dim == 1:
            plot_1D_complex_field(domain, psi, r"\psi_0", ax)
        else:
            plot_2D_complex_field(mesh, psi.reshape(mesh[0].shape), ax)
        
        plt.show()