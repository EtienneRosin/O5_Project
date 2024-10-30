# from schrodinger_engine/utils/graphics/complex_fields/multicolored_line_2D.py

from schrodinger_engine.utils.graphics.wave_function.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d
from schrodinger_engine.utils.domain import Domain

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
 
"""
TODO Implement the real and imag part to the plot_1D_complex_field function
"""

def plot_1D_complex_field(
    domain: Domain, 
    field: np.ndarray, 
    part: str|list[str] = None, 
    field_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender',
    type: str = 'surface') -> tuple:
    r"""Plot a 1D complex field where the values are the squared absolute values of the field and the coloring the phase.

    Parameters
    ----------
    domain: Domain
        Domain where the field is defined.
    field: np.ndarray
        Considered field
    part: str, optional
        Part to display with the current view (e.g. 'real' or 'imag'). Defaults to None.
    field_name: str|list[str], optional
        Label of the field. Defaults to '\psi(x)'.
    ax: plt.Axes, optional
        Axe onto which to plot the field. Defaults to None (meaning that a new figure will be created).
    cmap: str, optional
        Colormap to use. Defaults to 'cmr.lavender'.
    type: str, default 'surface'
        Type of plot. For the 1D case its a simple plot. For the 2D case it can be 'surface' or 'pcolormesh' (in this case the color is the phase and the transparency is the probability density).

    Returns
    -------
    ax: plt.Axes
        Axe onto which to plot the field.
    artists: list
        List of the created artists.
    """
    line_props = dict(alpha = 0.25, linewidth = 2)
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    module_label = r"\left|" + field_name + r"\right|^2"
    phase_label = r"arg\left(" + field_name + r"\right)"
    real_label = r"\Re\left(" + field_name + r"\right)"
    imag_label = r"\Im\left(" + field_name + r"\right)"
    mesh = domain.get_mesh()
    module = np.abs(field)**2
    phase = np.angle(field)
    
    color_map = mpl.colormaps[cmap]
    colors = color_map(np.linspace(0, 1, 2))
    
    norm = plt.Normalize(-np.pi, np.pi)
    line = MulticolorLine2d(
        x = mesh, 
        y = module,
        z = phase,
        cmap = cmap,
        norm = norm, 
        label = f"${module_label}$",
        linewidth = 4
        )
    artists = [line]
    ax.add_collection(line)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad="2%")
    plt.colorbar(line, cax=cax, label = f"${phase_label}$")
 
    if isinstance(part, str):
        match part:
            case 'real':
                real_line, = ax.plot(mesh, field.real, label=f"${real_label}$", c = colors[0], **line_props)
                artists.append(real_line)
            case 'imag':
                imag_line, = ax.plot(mesh, field.imag, label=f"${imag_label}$", c = colors[1], **line_props)
                artists.append(imag_line)
            case _:
                raise ValueError("Part should be 'real', 'imag', or both.")
    elif isinstance(part, list):
        for p in part:
            match p:
                case 'real':
                    real_line, = ax.plot(mesh, field.real, label=f"${real_label}$", c = colors[0], **line_props)
                    artists.append(real_line)
                case 'imag':
                    imag_line, = ax.plot(mesh, field.imag, label=f"${imag_label}$", c = colors[1], **line_props)
                    artists.append(imag_line)
                case _:
                    raise ValueError("Part should be 'real', 'imag', or both.")
    min_val = module.min()
    max_val = module.max()
    for artist in artists[1:]:
        data = artist.get_data()
        min = np.min(data)
        max = np.max(data)
        if min < min_val:
            min_val = min
        if max > max_val:
            max_val = max    
    ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
    ax.set(xlabel = r"$x$", xlim = (domain.boundaries[0]), ylim = (min_val*1.1, max_val*1.1))
    return ax, line

if __name__ == '__main__':
    spatial_domain = Domain(boundaries=[[0, 2]], N = 1000)
    mesh = spatial_domain.get_mesh()
    field = 2*np.cos(10*mesh)*np.exp(1j*100*mesh)
    
    ax, artists = plot_1D_complex_field(
        domain = spatial_domain, 
        field = field, 
        part=["real", "imag"]
        )
    # ax.legend()
    # print(ax.artists)
    plt.show()
    
    