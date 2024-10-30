from schrodinger_engine.utils.domain import Domain
from schrodinger_engine.utils.graphics.wave_function.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cmasher as cmr
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_wave_function(
    domain: Domain, 
    wave_function: np.ndarray, 
    prop_to_display: str|list[str] = 'density_probability', 
    wave_function_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender') -> tuple:
    r"""Plot a wave function defined over the given domain.

    Parameters
    ----------
    domain: Domain
        Domain where the wave_function is defined.
    wave_function: np.ndarray
        Considered wave_function
    prop_to_display: str, default 'density_probability'
        property to display with the current view (e.g. 'real', 'imag' or 'density_probability').
    wave_function_name: str|list[str], optional
        Label of the wave_function. Defaults to '\psi(x)'.
    ax: plt.Axes, default None
        Axe onto which to plot the wave_function. If None a new figure will be created.
    cmap: str, default 'cmr.lavender'
        Colormap to use.
    type: str, default 'surface'
        Type of plot. For the 1D case its a simple plot. For the 2D case it can be 'surface' or 'pcolormesh' (in this case the color is the phase and the transparency is the probability density).

    Returns
    -------
    ax: plt.Axes
        Axe onto which to plot the wave_function.
    artists: list
        List of the created artists.
    """
    # General properties --------------------------------------------
    mesh = domain.get_mesh()
    if domain.dim == 2:
        wave_function = wave_function.reshape(mesh[0].shape)
        
    probability_density = np.abs(wave_function)**2
    phase = np.angle(wave_function)
    
    module_label = r"\left|" + wave_function_name + r"\right|^2"
    phase_label = r"arg\left(" + wave_function_name + r"\right)"
    real_label = r"\Re\left(" + wave_function_name + r"\right)"
    imag_label = r"\Im\left(" + wave_function_name + r"\right)"
    
    color_map = mpl.colormaps['cmr.lavender']
    real_color, imag_color = color_map(np.linspace(0, 1, 2))
    phase_norm = plt.Normalize(-np.pi, np.pi)
    
    cb_ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
    cb_ticks_label = [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
    
    artists = []
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d' if domain.dim == 2 else None)
    else:
        fig = ax.figure
    
    # Plot the artists ----------------------------------------------
    if domain.dim == 1:
        line = MulticolorLine2d(
            x = mesh, 
            y = probability_density,
            z = phase,
            cmap = cmap,
            norm = phase_norm, 
            label = f"${module_label}$",
            linewidth = 4
            )
        ax.add_collection(line)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2.5%", pad="2%")
        cbar = plt.colorbar(line, cax=cax, label = f"${phase_label}$", ticks = cb_ticks)
        cbar.ax.set_yticklabels(cb_ticks_label)
        artists.append(line)
        
    
    
    
    
    return ax, artists
    pass

if __name__ == '__main__':
    spatial_domain = Domain(boundaries=[[0, 2]], N = 1000)
    mesh = spatial_domain.get_mesh()
    wave_function = 2*np.cos(10*mesh)*np.exp(1j*100*mesh)
    
    ax, artists = plot_wave_function(
        domain = spatial_domain, 
        wave_function = wave_function, 
        prop_to_display=["real", "imag"]
        )
    # ax.legend()
    # print(ax.artists)
    plt.show()