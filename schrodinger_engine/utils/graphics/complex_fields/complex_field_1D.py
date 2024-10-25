# from schrodinger_engine/utils/graphics/complex_fields/multicolored_line_2D.py

from schrodinger_engine.utils.graphics.complex_fields.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d
from schrodinger_engine.domains.test_domain import Domain

import cmasher as cmr
import numpy as np
import matplotlib.pyplot as plt

"""
TODO Implement the real and imag part to the plot_1D_complex_field function
"""

def plot_1D_complex_field(
    domain: Domain, 
    field: np.ndarray, 
    part: str = None, 
    field_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender') -> tuple[plt.axes, MulticolorLine2d]:
    r"""Plot a 1D complex field where the values are the squared absolute values of the field and the coloring the phase.

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
    line: MulticolorLine2d
        Created MulticolorLine2d object.
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
    line = MulticolorLine2d(
        x = domain.get_mesh(), 
        y = module, 
        z = phase, 
        cmap = cmap, 
        norm = norm, 
        label = f"${module_label}$"
        )
    
    ax.add_collection(line)
    return ax, line
    # plt.show()

if __name__ == '__main__':
    spatial_domain = Domain(boundaries=[[0, 1]], N = 100)
    field = np.cos(10*spatial_domain.get_mesh())*np.exp(1j*100*spatial_domain.get_mesh())
    # field = np.cos(spatial_domain.get_mesh())
    # print(field)
    
    ax, line = plot_1D_complex_field(domain=spatial_domain, field=field)
    # ax.legend()
    ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
    plt.show()
    
    