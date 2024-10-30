from schrodinger_engine.utils.graphics.wave_function import MulticolorLine2d, HandlerMulticolorLine2d, plot_1D_complex_field
from schrodinger_engine.utils.graphics.design import MSColors

from schrodinger_engine.utils.domain import Domain

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from abc import ABC, abstractmethod


# class InitialCondition(ABC):
#     """
#     Base class for InitialCondition. Must be subclassed to be usable.
#     """
#     @abstractmethod
#     def __call__(self, x):
#         """
#         Evaluate the initial condition on point(s) x.
        
#         Parameters
#         ----------
#         x : array-like
#             point(s) onto which evaluate the potential
            
#         Returns
#         -------
#         array-like
#             V(x): potential at x
#         """
#         pass
    
#     def display(self, ax = None, x_range = (-10, 10), num_points = 400):
#         """
#         Display the initial condition
        
#         Parameters
#         ----------
#         ax : Axes
#             Axis object on which to display the potential.
#         x_range : array-like
#             boundaries of the interval onto which display the potential
#         num_points : int
#             number of points of the discretization.
#         """
#         if ax is None:
#             fig = plt.figure()
#             ax = fig.add_subplot()
#             fig.canvas.manager.set_window_title(self.__class__.__name__)
#             ax.set(xlabel = r"$x$ []")
    
    
#         lst_x = np.linspace(x_range[0], x_range[1], num_points)  
        
#         values = self.__call__(lst_x) 
#         mag = np.abs(values)**2
#         phase = np.angle(values)
            
#         cmap = "cmr.iceburn"
#         # cmap = "cmr.guppy"
#         norm = plt.Normalize(-np.pi, np.pi)
#         line = MulticolorLine2d(x = lst_x, y = mag, z = phase, cmap=cmap, norm=norm, label = r"$\left|\psi_0\right|$")
#         ax.add_collection(line)

#         ax.plot(lst_x, values.real, label = r"$\Re\left(\psi_0\right)$", c = "blue", alpha = 0.5)
#         ax.plot(lst_x, values.imag, label = r"$\Im\left(\psi_0\right)$", c = "red", alpha = 0.5)
#         legend = ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
        
#         fig.colorbar(line, label= r"$arg\left(\psi_0\right)$")


class InitialCondition(ABC):
    """
    Base class for InitialCondition. 
    NOTE Must be subclassed to be usable.
    """
    @abstractmethod
    def __call__(self, x):
        r"""
        Evaluate the initial condition at point(s) x.
        
        Parameters
        ----------
        x : array-like
            point(s) onto which evaluate the potential
            
        Returns
        -------
        array-like
            \psi(x): potential at x
        """
        pass
    
    def display(self, ax = None, domain: Domain = None, cmap: str = 'cmr.lavender', part: str|list[str] = None):
        """
        Display the initial condition
        
        Parameters
        ----------
        ax : Axes
            Axis object on which to display the initial condition.
        domain: Domain
            Domain onto which to plot the initial condition.
        """
        
        if domain is None:
            boundaries = [[-40, 40] for _ in range(self.dim)]
            N = [1000 for _ in range(self.dim)]
            domain = Domain(boundaries = boundaries, N = N)
        
        mesh = domain.get_mesh()
        psi_0 = self.__call__(mesh)
        # color_map = mpl.colormaps[cmap]
        # colors = color_map(np.linspace(0, 1, 2))

        
        if ax is None:
            fig = plt.figure()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
            ax = fig.add_subplot(projection = "3d" if self.dim == 2 else None)

        if self.dim == 1 :
            ax, artists = plot_1D_complex_field(domain=domain, field=psi_0, field_name=r"\psi_0", ax=ax)
            plt.tight_layout()        
        
        else :
            psi_0 = psi_0.reshape(mesh[0].shape)
            phase = np.angle(psi_0)
            color_map = mpl.colormaps[cmap]
            
            phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
            color_map = mpl.colormaps[cmap]
            face_colors = color_map(phase_norm(phase)) 
            
            
            ax.plot_surface(
                *mesh, 
                np.abs(psi_0)**2,
                cmap = cmap,
                alpha = 0.75,
                # rstride=1, 
                # cstride=1, 
                facecolors = face_colors,
                # rstride=10, cstride=10,
                linewidth=0,
                antialiased=True, 
                shade=False
                # linewidth=0, 
                # antialiased=False
                )
            
            # Plot the 3D surface
            # ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
            #                 alpha=0.3)

            # Plot projections of the contours for each dimension.  By choosing offsets
            # that match the appropriate axes limits, the projected contours will sit on
            # the 'walls' of the graph.
            
            # ax.contour(
            #     *mesh, 
            #     phase,
            #     zdir='z', 
            #     offset=-1,
            #     cmap = cmap
            #     )
            # print(f"{domain.boundaries[0,0] = }")
            
            # ax.contour(
            #     *mesh, 
            #     psi_0.real, 
            #     zdir='x', 
            #     offset= 1.1 * domain.boundaries[0,0],
            #     cmap = cmap
            #     )
            # ax.contour(
            #     *mesh, 
            #     psi_0.real, 
            #     zdir='y', 
            #     offset= 1.1 * domain.boundaries[1,1],
            #     # cmap='coolwarm'
            #     cmap = cmap
            #     )

            # ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
            #     xlabel='X', ylabel='Y', zlabel='Z')
        

if __name__ == "__main__":
    pass