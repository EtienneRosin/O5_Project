from schrodinger_engine.visualization_tools import MulticolorLine2d, HandlerMulticolorLine2d

import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod


class InitialCondition(ABC):
    """
    Base class for InitialCondition. Must be subclassed to be usable.
    """
    @abstractmethod
    def __call__(self, x):
        """
        Evaluate the initial condition on point(s) x.
        
        Parameters
        ----------
        x : array-like
            point(s) onto which evaluate the potential
            
        Returns
        -------
        array-like
            V(x): potential at x
        """
        pass
    
    def display(self, ax = None, x_range = (-10, 10), num_points = 400):
        """
        Display the initial condition
        
        Parameters
        ----------
        ax : Axes
            Axis object on which to display the potential.
        x_range : array-like
            boundaries of the interval onto which display the potential
        num_points : int
            number of points of the discretization.
        """
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
            ax.set(xlabel = r"$x$ []")
    
    
        lst_x = np.linspace(x_range[0], x_range[1], num_points)  
        
        values = self.__call__(lst_x) 
        mag = np.abs(values)**2
        phase = np.angle(values)
            
        cmap = "cmr.iceburn"
        # cmap = "cmr.guppy"
        norm = plt.Normalize(-np.pi, np.pi)
        line = MulticolorLine2d(x = lst_x, y = mag, z = phase, cmap=cmap, norm=norm, label = r"$\left|\psi_0\right|$")
        ax.add_collection(line)

        ax.plot(lst_x, values.real, label = r"$\Re\left(\psi_0\right)$", c = "blue", alpha = 0.5)
        ax.plot(lst_x, values.imag, label = r"$\Im\left(\psi_0\right)$", c = "red", alpha = 0.5)
        legend = ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
        
        fig.colorbar(line, label= r"$arg\left(\psi_0\right)$")

if __name__ == "__main__":
    pass