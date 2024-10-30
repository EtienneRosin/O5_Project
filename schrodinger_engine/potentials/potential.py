

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union
from schrodinger_engine.utils.domain import Domain
from schrodinger_engine.utils.graphics.wave_function import prepare_points_for_pcolormesh

from matplotlib.animation import FuncAnimation

# class Potential1D(ABC):
#     """
#     Base class for for time-independent potentials. Must be subclassed to be usable.
#     """
#     @abstractmethod
#     def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
#         """
#         Evaluate the potential on point(s) x
        
#         Parameters
#         ----------
#         x : array-like
#             Point(s) onto which evaluate the potential.
        
#         Returns
#         -------
#         array-like
#             V(x): potential at x.
#         """
#         pass
    
#     def display(self, ax: plt.axes = None, x_range=(-15, 15), num_points=2000):
#         """
#         Display the potential on a domain.
        
#         Parameters
#         ----------
#         ax : Axis
#             Axis object on which to display the potential.
#         """
#         if ax is None:
#             fig = plt.figure()
#             ax = fig.add_subplot()
#             fig.canvas.manager.set_window_title(self.__class__.__name__)
#             ax.set(xlabel = r"$x$ []")
#         lst_x = np.linspace(x_range[0], x_range[1], num_points)
        
#         values = self.__call__(lst_x)
        
#         ax_props = dict(
#             xlabel = r"$x$ []", 
#             ylabel = r"$V(x)$ []", 
#             # aspect = "equal"
#         )
#         ax.set(**ax_props)
#         ax.plot(lst_x, values)
        

class Potential(ABC):
    """
    Base class for potentials.
    NOTE Must be subclassed to be usable.
    """
    @abstractmethod
    def __call__(self, x: np.ndarray) -> float|np.ndarray:
        """
        Evaluate the potential on point(s) x
        
        Parameters
        ----------
        x : np.ndarray
            Point(s) onto which evaluate the potential.
        
        Returns
        -------
        array-like
            V(x): potential at x.
        """
        pass
    
    def _validate_dim(self, dim: int) -> int:
        """
        Validates the dimension.

        Args:
            dim: int
                dimension

        Returns:
            int
            validated dimension
        """
        if dim not in [1, 2]:
            raise ValueError("Dimension should be either 1 or 2")
        return dim
    
    def display(self, domain: Domain, ax: plt.axes = None):
        """
        Display the potential on a domain.
        
        Parameters
        ----------
        domain: Domain
            Considered domain
        ax : Axis
            Axis object on which to display the potential.
        """
        if ax is None:
            fig = plt.figure()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
            
            
            mesh = domain.get_mesh()
            
            V = self.__call__(mesh)
            match self.dim :
                case 1:
                    ax = fig.add_subplot()
                    ax.set(xlabel = r"$x$", ylabel = r"$V(x)$")
                    ax.plot(domain.get_mesh(), V)
                case 2:
                    ax = fig.add_subplot(projection = '3d')
                    ax.set(xlabel = r"$x$", ylabel = r"$y$", zlabel = r"$V(x, y)$")
                    # X, Y, Z = prepare_points_for_pcolormesh(*domain.unique_points, V)
                    Z = V.reshape(mesh[0].shape)
                    ax.plot_surface(*mesh, Z)
                    # ax.plot_surface(*domain.get_mesh(), V)
                
            
            # ax.set(xlabel = r"$x$ []")
        # lst_x = np.linspace(x_range[0], x_range[1], num_points)
        
        
        
        # ax_props = dict(
        #     xlabel = r"$x$ []", 
        #     ylabel = r"$V(x)$ []", 
        #     # aspect = "equal"
        # )
        # ax.set(**ax_props)
        # ax.plot(lst_x, values)