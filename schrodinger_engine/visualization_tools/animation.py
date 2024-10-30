import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmasher as cmr

from schrodinger_engine.utils.graphics.wave_function import MulticolorLine2d, multicolored_line_2D

class SimulationReader:
    def __init__(
        self, file_path: str, 
        fps: int = 120,
        dpi: int = 500):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, "r")
        
        

        self.spatial_mesh = self.hdf5_file["spatial_mesh"][:]
        self.temporal_mesh = self.hdf5_file["temporal_mesh"][:]
        self.num_iterations = len(self.temporal_mesh)
        self.simulation_type = self.hdf5_file.attrs.get("simulation_type", "Unknown")
        self.dt = self.hdf5_file.attrs.get("dt", "Unknown")
        
        self.wave_function_props = {
            'real' : {'label': r"\Re\left(\psi\right)", 'func': lambda x: x.real},
            'imag' : {'label': r"\Im\left(\psi\right)", 'func': lambda x: x.imag},
            'probability_density' : {'label': r"\left|\psi\right|^2", 'func': lambda x: np.abs(x)**2},
            'phase' : {
                'label': r"\arg\left(\psi\right)", 
                'func': lambda x: np.angle(x),
                'ticks': [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
                'ticks_label': [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]}
        }
        
        self.animation_props = dict(fps = int(1/self.dt), dpi = dpi, writer = 'ffmpeg')
        self.animation_props = dict(fps = fps, dpi = dpi, writer = 'ffmpeg')
        self.fps = fps
        
        
    def _prepare_figure(self, prop_to_display):
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d' if self.spatial_mesh.ndim == 3 else None)
        plt.tight_layout()
        
        ax_prop = self._get_bounding_box(prop_to_display=prop_to_display)
        ax_prop.update({'xlabel': r"$x$"})
        if self.spatial_mesh.ndim == 3:
            ax_prop.update({'ylabel': r"$y$", 'zlabel': f"${self.wave_function_props[prop_to_display]['label']}$"})
        else:
            ax_prop.update({'ylabel': f"${self.wave_function_props[prop_to_display]['label']}$"})
        
        ax.set(**ax_prop)
        return fig, ax
    
    def _get_bounding_box(self, prop_to_display):
        bbox_dict = {}
        
        psi = self._get_wave_function(iteration=self.num_iterations//4)
        psi_prop = self.wave_function_props[prop_to_display]['func'](psi)
        match prop_to_display:
            case 'real':
                prop_min = psi_prop.min()
                prop_max = psi_prop.max()
            case 'imag':
                prop_min = psi_prop.min()
                prop_max = psi_prop.max()
            case 'probability_density':
                prop_max = psi_prop.max()
                prop_min = -0.05*prop_max
            case 'phase':
                prop_min = -np.pi
                prop_max = np.pi
        
        if self.spatial_mesh.ndim == 3:
            bbox_dict.update({'xlim' : (self.spatial_mesh[0].min(), self.spatial_mesh[0].max())})
            bbox_dict.update({'ylim' : (self.spatial_mesh[1].min(), self.spatial_mesh[1].max())})
            bbox_dict.update({'zlim' : (prop_min, prop_max)})

        else:
            bbox_dict.update({'xlim' : (self.spatial_mesh.min(), self.spatial_mesh.max())})
            bbox_dict.update({'ylim' : (prop_min, prop_max)})
        return bbox_dict    
    
    def _get_wave_function(self, iteration):
        """Retourne la fonction d'onde pour une itération spécifique."""
        if iteration < 0 or iteration >= self.num_iterations:
            raise IndexError("Iteration out of range.")
        return self.hdf5_file["wave_function"][iteration]

    def animate_solution(
        self, 
        prop_to_display: str = 'probability_density', 
        cmap: str = 'cmr.lavender', 
        save_name: str = None,
        file_format: str = 'mp4'):
        """Visualisation dynamique de la fonction d'onde au cours du temps."""
        fig, ax = self._prepare_figure(prop_to_display = prop_to_display)
        
        
        if self.spatial_mesh.ndim == 1:
            line_wave_function, = ax.plot([], [], lw=2)
            
            if self.simulation_type == 'stationnary':
                line_potential, = ax.plot([], [], lw=1, color = "black", alpha = 0.25, label = r"$V/\|V\|$")

            def init():
                line_wave_function.set_data([], [])
                
                # line_potential.set_data(self.spatial_mesh, self.hdf5_file["potential"][:]/np.linalg.norm(self.hdf5_file["potential"][:]))
                
                if self.simulation_type == 'stationnary':
                    line_potential.set_data(self.spatial_mesh, self.hdf5_file["potential"][:]/np.linalg.norm(self.hdf5_file["potential"][:]))
                    ax.legend()
                    return line_wave_function,
                return line_wave_function,

            def update(frame):
                psi = self._get_wave_function(frame)
                line_wave_function.set_data(self.spatial_mesh, self.wave_function_props[prop_to_display]['func'](psi))
                ax.set_title(f"$t$ = {self.temporal_mesh[frame]:.2f}")
                if self.simulation_type == 'stationnary':
                    return line_wave_function, line_potential,
                else:
                    return line_wave_function,
        else:
            surface_props = dict(
            alpha = 0.85,
            linewidth=0, 
            antialiased=True,
            zorder = 2,
            shade=False
            )
            
            psi = self._get_wave_function(0)
            
            plot = [ax.plot_surface(
                        *self.spatial_mesh,
                        self.wave_function_props[prop_to_display]['func'](psi).reshape(self.spatial_mesh[0].shape),
                        cmap = cmap,
                        **surface_props)
                    ]
            # if self.simulation_type == 'stationnary':
            #     V = self.hdf5_file["potential"][:]/np.linalg.norm(self.hdf5_file["potential"][:])
            #     V = V.reshape(self.spatial_mesh[0].shape)
            #     # V = np.where(V != 0, V, np.nan)
                                                                  
            #     plot.append(ax.plot_surface(
            #             *self.spatial_mesh,
            #             V,
            #             color = 'black',
            #             alpha = 0.25,
            #             linewidth=0, 
            #             antialiased=True,
            #             shade = False))
            def init():
                psi = self._get_wave_function(0)
                plot[0].remove()
                
                plot[0] = ax.plot_surface(
                        *self.spatial_mesh,
                        self.wave_function_props[prop_to_display]['func'](psi).reshape(self.spatial_mesh[0].shape),
                        cmap = cmap,
                        **surface_props)
                
                return plot,
            
            def update(frame):
                psi = self._get_wave_function(frame)
                plot[0].remove()
                
                plot[0] = ax.plot_surface(
                        *self.spatial_mesh,
                        self.wave_function_props[prop_to_display]['func'](psi).reshape(self.spatial_mesh[0].shape),
                        cmap = cmap,
                        **surface_props
                        )
                ax.set_title(f"$t$ = {self.temporal_mesh[frame]:.2f}")
                return plot,

        ani = animation.FuncAnimation(
            fig = fig, 
            func = update, 
            frames = self.num_iterations, 
            init_func = init,
            # interval = 1000/self.fps
            interval = 1000*self.dt
            )
        if save_name:
            ani.save(filename = f"{save_name}.{file_format}", **self.animation_props)
            
        plt.show()

    def close(self):
        """Ferme le fichier HDF5 pour libérer les ressources."""
        self.hdf5_file.close()
    
    def __del__(self):
        self.close()

if __name__ == '__main__':
    # Remplacez le chemin par celui de votre fichier HDF5 de simulation
    file_path = "results/simulation.h5"
    reader = SimulationReader(file_path)
    
    print(reader.simulation_type)
    # print(reader.hdf5_file["potential"])
    reader.animate_solution(
        prop_to_display='real',
        # save_name = 'test_2D_proba'
        )
    
    reader.close()