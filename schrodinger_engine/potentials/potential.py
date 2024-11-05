"""
This module defines the abstract base classes `Potential` and `TimeDependentPotential` 
for representing and visualizing static and time-dependent quantum mechanical potentials 
in 1D or 2D spatial domains.

Notes
-----
These classes are abstract and must be subclassed with specific implementations of 
the potential function to be used in applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from schrodinger_engine.utils.domain import Domain
from matplotlib.animation import FuncAnimation
import cmasher as cmr

# 2D surface plot configuration
surface_props = dict(
    alpha=0.85, linewidth=0, antialiased=True, zorder=2, shade=False
)


class Potential(ABC):
    """
    Abstract base class for representing static potentials in 1D or 2D quantum mechanics.

    Attributes
    ----------
    dim : int
        Dimension of the potential (1 or 2).
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> float | np.ndarray:
        """
        Evaluate the potential at the given point(s).

        Parameters
        ----------
        x : np.ndarray
            Coordinates at which to evaluate the potential.

        Returns
        -------
        float or np.ndarray
            The potential evaluated at the input coordinates.
        """
        pass

    def _validate_dim(self, dim: int) -> int:
        """
        Validates that the dimension is either 1 or 2.

        Parameters
        ----------
        dim : int
            The specified dimension for the potential.

        Returns
        -------
        int
            The validated dimension.

        Raises
        ------
        ValueError
            If the dimension is not 1 or 2.
        """
        if dim not in [1, 2]:
            raise ValueError("Dimension should be either 1 or 2.")
        return dim

    def display(self, domain: Domain, ax: plt.axes = None):
        """
        Display the static potential over a specified domain.

        Parameters
        ----------
        domain : Domain
            Domain specifying the spatial boundaries and resolution.
        ax : matplotlib.axes.Axes, optional
            Axes on which to display the potential. If None, a new figure and axes 
            are created.

        Notes
        -----
        In 1D, displays a line plot. In 2D, displays a surface plot.
        """
        if ax is None:
            fig = plt.figure()
            fig.canvas.manager.set_window_title(self.__class__.__name__)

        mesh = domain.get_mesh()
        V = self.__call__(mesh)

        match self.dim:
            case 1:
                ax = fig.add_subplot()
                ax.plot(mesh, V)
                ax.set(xlabel=r"$x$", ylabel=r"$V(x)$", aspect='equal')
            case 2:
                ax = fig.add_subplot(projection='3d')
                Z = V.reshape(mesh[0].shape)
                ax.plot_surface(*mesh, Z, cmap = 'cmr.lavender', **surface_props)
                ax.set(xlabel=r"$x$", ylabel=r"$y$", zlabel=r"$V(x, y)$", aspect='equal')

class TimeDependentPotential(Potential):
    """
    Abstract base class for representing time-dependent potentials in 1D or 2D.

    Extends `Potential` to support time-dependent potential evaluation and visualization.
    """

    @abstractmethod
    def __call__(self, x: float | np.ndarray, t: float) -> float | np.ndarray:
        """
        Evaluate the potential at specified point(s) and time.

        Parameters
        ----------
        x : array-like
            Coordinates at which to evaluate the potential.
        t : float
            Time at which to evaluate the potential.

        Returns
        -------
        array-like
            The potential evaluated at the input coordinates and time.
        """
        pass
    
    def _prepare_figure(self):
        """
        Prepare the figure and axes for displaying the potential.

        Returns
        -------
        tuple
            The figure and axis objects.
        """
        fig = plt.figure()
        fig.canvas.manager.set_window_title(self.__class__.__name__)
        ax = fig.add_subplot(projection='3d' if self.dim == 2 else None)
        plt.tight_layout()
        
        ax_props = {'xlabel': r"$x$"}
        if self.dim == 2:
            ax_props.update({'ylabel': r"$y$", 'zlabel': f"$V(x,y,t)$"})
        else:
            ax_props.update({'ylabel': f"$V(x,t)$"})
        
        ax.set(**ax_props)
        return fig, ax
    
    def display(self, spatial_domain: Domain, temporal_domain: Domain):
        """
        Display an animated visualization of the time-dependent potential over specified domains.

        Parameters
        ----------
        spatial_domain : Domain
            Domain specifying the spatial boundaries and resolution.
        temporal_domain : Domain
            Domain specifying the time boundaries and resolution.
        
        Notes
        -----
        In 1D, displays an animated line plot showing V(x, t). In 2D, displays an animated 
        surface plot showing V(x, y, t).
        """
        fig, ax = self._prepare_figure()
        mesh = spatial_domain.get_mesh()
        lst_t = temporal_domain.get_mesh()
        dt = temporal_domain.step[0]
        
        V = self.__call__(mesh, lst_t)
        v_min, v_max = V.min(), V.max()

        if self.dim == 1:
            ax.set(ylim=(v_min, v_max))
            line, = ax.plot([], [], lw=2)

            def init():
                line.set_data([], [])
                return line,

            def update(frame):
                line.set_data(mesh, V[frame])
                ax.set_title(f"$t$ = {lst_t[frame]:.2f}")
                return line,
        else:
            
            ax.set(zlim=(v_min, v_max))
            plot = [ax.plot_surface(
                *mesh, V[0].reshape(mesh[0].shape), cmap='cmr.lavender', **surface_props
            )]

            def init():
                plot[0].remove()
                plot[0] = ax.plot_surface(
                    *mesh, V[0].reshape(mesh[0].shape), cmap='cmr.lavender', **surface_props
                )
                return plot,

            def update(frame):
                plot[0].remove()
                plot[0] = ax.plot_surface(
                    *mesh, V[frame].reshape(mesh[0].shape), cmap='cmr.lavender', **surface_props
                )
                ax.set_title(f"$t$ = {lst_t[frame]:.2f}")
                return plot,

        ani = FuncAnimation(
            fig=fig, func=update, frames=range(len(lst_t)), init_func=init, interval=1000 * dt
        )
        plt.show()