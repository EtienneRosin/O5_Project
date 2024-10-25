# from schrodinger_engine/utils/graphics/complex_fields/multicolored_line_2D.py

from schrodinger_engine.utils.graphics.complex_fields.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d
from schrodinger_engine.domains.test_domain import Domain

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

"""
TODO Implement the real and imag part to the plot_1D_complex_field function
"""

def plot_2D_complex_field_as_pcolormesh(
    domain: Domain, 
    field: np.ndarray, 
    part: str = None, 
    field_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender') -> tuple[plt.axes, mpl.collections.QuadMesh]:
    r"""Plot a 2D complex field as a pcolormesh where the values are the squared absolute values of the field and the coloring the phase.

    Parameters
    ----------
    domain: Domain)
        Domain where the field is defined.
    field: np.ndarray
        Considered field
    part: str, optional
        Part to display with the current view (e.g. 'real' or 'imag'). Defaults to None.
    field_name: str, optional
        Label of the field. Defaults to '\psi(x)'.
    ax: plt.Axes, optional
        Axe onto which to plot the field. Defaults to None (meaning that a new figure will be created).
    cmap: str, optional
        Colormap to use. Defaults to 'cmr.lavender'.

    Returns
    -------
    ax: plt.Axes
        Axe onto which to plot the field.
    pcm: mpl.collections.QuadMesh
        Created pcolormesh.
    """
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    module_label = r"\left|" + field_name + r"\right|^2"
    # phase_label = r"arg\left(" + field_name + r"\right)"
    module = np.abs(field)**2
    phase = np.angle(field)
    
    # cmap = "cmr.lavender"
    
    norm = plt.Normalize(-np.pi, np.pi)
    # line = MulticolorLine2d(
    #     x = domain.get_mesh(), 
    #     y = module, 
    #     z = phase, 
    #     cmap = cmap, 
    #     norm = norm, 
    #     label = f"${module_label}$"
    #     )
    pcm = ax.pcolormesh(*domain.get_mesh(), phase, cmap = cmap)
    # im = ax.imshow(X = module, cmap = cmap, extent = domain.boundaries.flatten(), vmin = module.min(), vmax=module.max())
    
    # ax.add_collection(line)
    return ax, pcm
    # plt.show()

if __name__ == '__main__':
    spatial_domain = Domain(boundaries=[[0, 10], [0, 10]], N = [400, 400])
    
    print(f"{spatial_domain.boundaries.flatten()}")
    k = np.array([1, 1])
    
    field = np.exp(-1j * np.tensordot(k, spatial_domain.get_mesh(), axes = (0,0)))
    # np.cos(10*spatial_domain.get_mesh())*
    print(field.shape)
    
    ax, im = plot_2D_complex_field_as_pcolormesh(domain=spatial_domain, field=field)
    plt.show()
    # ax, line = plot_1D_complex_field(domain=spatial_domain, field=field)
    # # ax.legend()
    # ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
    # plt.show()
    
    