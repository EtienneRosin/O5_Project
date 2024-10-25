"""
@file potentials.py
@brief Implementation of an abstract class that handles different potentials
@author Etienne Rosin 
@version 0.1
@date 28/09/2024
"""
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Union

from matplotlib.animation import FuncAnimation

class Potential1D(ABC):
    """
    Base class for for time-independent potentials. Must be subclassed to be usable.
    """
    @abstractmethod
    def __call__(self, x: float | np.ndarray) -> float | np.ndarray:
        """
        Evaluate the potential on point(s) x
        
        Parameters
        ----------
        x : array-like
            Point(s) onto which evaluate the potential.
        
        Returns
        -------
        array-like
            V(x): potential at x.
        """
        pass
    
    def display(self, ax: plt.axes = None, x_range=(-15, 15), num_points=2000):
        """
        Display the potential on a domain.
        
        Parameters
        ----------
        ax : Axis
            Axis object on which to display the potential.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
            ax.set(xlabel = r"$x$ []")
        lst_x = np.linspace(x_range[0], x_range[1], num_points)
        
        values = self.__call__(lst_x)
        
        ax_props = dict(
            xlabel = r"$x$ []", 
            ylabel = r"$V(x)$ []", 
            # aspect = "equal"
        )
        ax.set(**ax_props)
        ax.plot(lst_x, values)
        
        
        
        
class TimeDependentPotential1D(Potential1D):
    """
    Base class for time-dependent potentials.
    """
    @abstractmethod
    def __call__(self, x: float | np.ndarray, t: float) -> float | np.ndarray :
        """
        Evaluate the potential on point(s) x at time t.
        
        Parameters
        ----------
        x : array-like
            Point(s) onto which evaluate the potential.
        t : float
            Time at which to evaluate the potential.
        
        Returns
        -------
        array-like
            V(x, t): potential at x and time t.
        """
        pass
    
    def display(self, ax: plt.axes = None, x_range=(-15, 15), num_points=200, T: float = 10) -> None:
        """
        Display the potential at a given time t.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
            ax.set(xlabel = r"$x$ []")
        
        lst_t = np.linspace(start = 0, stop = T, num = 100, endpoint = True)
        lst_x = np.linspace(x_range[0], x_range[1], num_points)
        values = self.__call__(lst_x, lst_t)
        
        potential_line, = ax.plot(lst_x, values[0, :])
        
        
        ylim = np.abs(values).max() * np.array([-1, 1])
        ax.set(xlabel=r"$x$", ylabel=r"$V(x, t)$", ylim = ylim, title=f"t = {lst_t[0]:.3g}")
        
        
        def update(frame):
            potential_line.set_ydata(values[frame, :])
            ax.set(title=f"t = {lst_t[frame]:.3g}")
            return potential_line, ax,


        # Create the animation
        ani = FuncAnimation(fig, update, frames = len(lst_t), blit=False, interval=50)
        plt.show()
        # ax.plot(lst_x, values)
        # ax.set(xlabel=r"$x$", ylabel=r"$V(x, t)$")



if __name__ == "__main__":
    pass
    # barriere = Wall(x_0 = 0, V_0 = 2, b = 0.5)
    # p = Potential()
    # p.__call__()
    
    
    # a = 2 + 1.66 *1j 
    # print(f"{a!r}")
    # barriere.display()
    
    # a = np.array(range(0, 4))
    # b = np.array(range(0, 4))
    # c = np.logical_or(a, b)
    # print(f"{c = }")