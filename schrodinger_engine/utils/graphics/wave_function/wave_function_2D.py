# # from schrodinger_engine/utils/graphics/complex_fields/multicolored_line_2D.py

# from schrodinger_engine.utils.graphics.complex_fields.multicolored_line_2D import MulticolorLine2d, HandlerMulticolorLine2d
from schrodinger_engine.utils.domain import Domain

# import cmasher as cmr
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl

# """
# TODO Implement the real and imag part to the plot_1D_complex_field function
# """

# def plot_2D_complex_field_as_pcolormesh(
#     domain: Domain, 
#     field: np.ndarray, 
#     part: str = None, 
#     field_name: str = r'\psi(x)', 
#     ax: plt.Axes = None,
#     cmap: str = 'cmr.lavender') -> tuple[plt.axes, mpl.collections.QuadMesh]:
#     r"""Plot a 2D complex field as a pcolormesh where the values are the squared absolute values of the field and the coloring the phase.

#     Parameters
#     ----------
#     domain: Domain)
#         Domain where the field is defined.
#     field: np.ndarray
#         Considered field
#     part: str, optional
#         Part to display with the current view (e.g. 'real' or 'imag'). Defaults to None.
#     field_name: str, optional
#         Label of the field. Defaults to '\psi(x)'.
#     ax: plt.Axes, optional
#         Axe onto which to plot the field. Defaults to None (meaning that a new figure will be created).
#     cmap: str, optional
#         Colormap to use. Defaults to 'cmr.lavender'.

#     Returns
#     -------
#     ax: plt.Axes
#         Axe onto which to plot the field.
#     pcm: mpl.collections.QuadMesh
#         Created pcolormesh.
#     """
    
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot()
#     module_label = r"\left|" + field_name + r"\right|^2"
#     # phase_label = r"arg\left(" + field_name + r"\right)"
#     mesh = domain.get_mesh()
#     field = field.reshape(mesh[0].shape)
#     module = np.abs(field)**2
#     phase = np.angle(field)
    
#     phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
#     color_map = mpl.colormaps[cmap]
    
#     # Create the RGBA color based on phase colormap and intensity
#     color_data = color_map(phase_norm(phase))
#     color_data[..., -1] = module / np.max(module)
#     pcm = ax.pcolormesh(*mesh, color_data, cmap = cmap)
#     ax.set(aspect = "equal")
#     return ax, pcm

# def plot_2D_complex_field_as_surf(
#     domain: Domain, 
#     field: np.ndarray, 
#     part: str = None, 
#     field_name: str = r'\psi(x)', 
#     ax: plt.Axes = None,
#     cmap: str = 'cmr.lavender'
#     ):
#     if ax is None:
#         fig = plt.figure()
#         ax = fig.add_subplot(projection = '3d')
    
#     field = field.reshape(mesh[0].shape)
#     phase = np.angle(field)
#     color_map = mpl.colormaps[cmap]
    
#     phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
#     color_map = mpl.colormaps[cmap]
#     face_colors = color_map(phase_norm(phase)) 
            
            
#     ax.plot_surface(
#         *mesh, 
#         np.abs(field)**2,
#         cmap = cmap,
#         alpha = 0.75,
#         # rstride=1, 
#         # cstride=1, 
#         facecolors = face_colors,
#         # rstride=10, cstride=10,
#         linewidth=0,
#         antialiased=True, 
#         shade=False
#         # linewidth=0, 
#         # antialiased=False
#         )

# if __name__ == '__main__':
#     spatial_domain = Domain(boundaries=[[0, 10], [0, 10]], N = [400, 400])
#     mesh = spatial_domain.get_mesh()
#     x_0 = 5 * np.ones(2)
#     sigma = 1
#     print(f"{spatial_domain.boundaries.flatten()}")
#     k = np.array([1, 1])
#     x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])
#     envelope = np.exp(-np.sum((x - x_0[:, np.newaxis])**2, axis=0) / (4 * sigma**2))

#     # Plane wave part
#     plane_wave = np.exp(1j * np.dot(k, x))
#     psi = envelope * plane_wave
#     # if x.size > 1:
#     psi /= np.linalg.norm(psi)
#     # psi.reshape(mesh[0].shape)
    
#     # field = np.exp(-1j * np.tensordot(k, spatial_domain.get_mesh(), axes = (0,0)))
#     # np.cos(10*spatial_domain.get_mesh())*
#     # print(field.shape)
    
#     ax, im = plot_2D_complex_field_as_pcolormesh(domain=spatial_domain, field=psi)
#     plt.show()
#     plot_2D_complex_field_as_surf(domain=spatial_domain, field=psi)
#     plt.show()
#     # ax, line = plot_1D_complex_field(domain=spatial_domain, field=field)
#     # # ax.legend()
#     # ax.legend(handler_map={MulticolorLine2d: HandlerMulticolorLine2d(numpoints=100)})
#     # plt.show()
    
    
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
from matplotlib.colors import Normalize
import cmasher as cmr

def plot_2D_complex_field_as_pcolormesh(
    domain: Domain, 
    field: np.ndarray, 
    part: str = None, 
    field_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender') -> tuple[plt.axes, mpl.collections.QuadMesh]:
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot()
    else:
        fig = ax.figure
    
    mesh = domain.get_mesh()
    field = field.reshape(mesh[0].shape)
    module = np.abs(field)**2
    phase = np.angle(field)
    
    phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    color_map = mpl.colormaps[cmap]
    
    color_data = color_map(phase_norm(phase))
    color_data[..., -1] = module / np.max(module)
    
    pcm = ax.pcolormesh(*mesh, color_data, cmap=cmap, shading='auto')
    ax.set(aspect="equal")
    
    # Ajouter une colorbar pour la phase
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=phase_norm, cmap=color_map), ax=ax)
    cbar.set_label(r"Phase (radians)", rotation=270, labelpad=15)
    
    return ax, pcm

def plot_2D_complex_field_as_surf(
    domain: Domain, 
    field: np.ndarray, 
    part: str = None, 
    field_name: str = r'\psi(x)', 
    ax: plt.Axes = None,
    cmap: str = 'cmr.lavender'):
    
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        fig = ax.figure
    
    mesh = domain.get_mesh()
    field = field.reshape(mesh[0].shape)
    phase = np.angle(field)
    magnitude_squared = np.abs(field)**2
    
    phase_norm = plt.Normalize(vmin=-np.pi, vmax=np.pi)
    color_map = mpl.colormaps[cmap]
    face_colors = color_map(phase_norm(phase))
    
    # Affichage de la surface
    surf = ax.plot_surface(
        *mesh, 
        magnitude_squared,
        facecolors=face_colors,
        # rstride=1, cstride=1,
        alpha = 0.75,
        linewidth=0, 
        antialiased=True, 
        shade=False
    )
    
    # Ajouter une colorbar pour la phase
    mappable = mpl.cm.ScalarMappable(norm=phase_norm, cmap=color_map)
    mappable.set_array(phase)
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label(r"Phase (radians)", rotation=270, labelpad=15)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel(r'$|\psi|^2$')
    
    return ax, surf

if __name__ == '__main__':
    spatial_domain = Domain(boundaries=[[0, 10], [0, 10]], N=[400, 400])
    mesh = spatial_domain.get_mesh()
    x_0 = 5 * np.ones(2)
    sigma = 1
    k = np.array([1, 1])
    x = np.vstack([mesh[0].ravel(), mesh[1].ravel()])
    envelope = np.exp(-np.sum((x - x_0[:, np.newaxis])**2, axis=0) / (4 * sigma**2))

    plane_wave = np.exp(1j * np.dot(k, x))
    psi = (envelope * plane_wave).reshape(mesh[0].shape)
    psi /= np.linalg.norm(psi)
    
    fig, ax = plt.subplots()
    plot_2D_complex_field_as_pcolormesh(domain=spatial_domain, field=psi, ax=ax)
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    plot_2D_complex_field_as_surf(domain=spatial_domain, field=psi, ax=ax)
    plt.show()