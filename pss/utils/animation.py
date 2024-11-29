import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import cmasher as cmr
import matplotlib.colors as colors

class SimulationReader:
    WAVE_FUNCTION_PROPERTIES = {
        'real': {'label': r"\Re\left(\psi\right)", 'func': lambda x: x.real},
        'imag': {'label': r"\Im\left(\psi\right)", 'func': lambda x: x.imag},
        'probability_density': {'label': r"\left|\psi\right|^2", 'func': lambda x: np.abs(x)**2},
        'phase': {
            'label': r"\arg\left(\psi\right)", 
            'func': lambda x: np.angle(x),
            'ticks': [-np.pi, -np.pi/2, 0, np.pi/2, np.pi],
            'ticks_label': [r"$-\pi$", r"$-\frac{\pi}{2}$", r"$0$", r"$\frac{\pi}{2}$", r"$\pi$"]
        }
    }
    
    def __init__(self, file_path: str, fps: int = 30, dpi: int = 500):
        self.file_path = file_path
        self.hdf5_file = h5py.File(file_path, "r")
        self._load_data()
        self.animation_props = {'fps': fps, 'dpi': dpi, 'writer': 'ffmpeg'}
        
    def _load_data(self):
        """Load simulation data from HDF5 file."""
        self.spatial_mesh = self.hdf5_file["spatial_mesh"][:]
        self.temporal_mesh = self.hdf5_file["temporal_mesh"][:]
        self.num_iterations = len(self.temporal_mesh)
        self.simulation_type = self.hdf5_file.attrs.get("simulation_type", "Unknown")
        self.dt = self.hdf5_file.attrs.get("dt", "Unknown")
        
    def _get_wave_function(self, iteration):
        """Get wave function at specific iteration."""
        if not 0 <= iteration < self.num_iterations:
            raise IndexError("Iteration out of range.")
        return self.hdf5_file["wave_function"][iteration]
    
    def _setup_1d_plot(self, ax, prop_to_display):
        """Setup 1D plot with given property."""
        lines = []
        lines.append(ax.plot([], [], lw=2)[0])
        
        if self.simulation_type == 'stationnary':
            potential_line = ax.plot(
                self.spatial_mesh, 
                self.hdf5_file["potential"][:]/np.linalg.norm(self.hdf5_file["potential"][:]),
                lw=1, color="black", alpha=0.25, label=r"$V/\|V\|$"
            )[0]
            lines.append(potential_line)
            ax.legend()
        ax.set(xlabel = r"$x$", ylabel = f"${self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['label']}$", xlim = (self.spatial_mesh.min(), self.spatial_mesh.max()))
        return lines
    
    def _setup_2d_plot(self, ax, prop_to_display, cmap):
        """Setup 2D plot with given property."""
        psi = self._get_wave_function(0)
        
        max_val = max(
            np.abs(self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['func'](self._get_wave_function(i))).max() 
            for i in range(self.num_iterations)
        )
            
        self.surface_props = {
            'alpha': 0.85,
            'linewidth': 0,
            'antialiased': True,
            'zorder': 2,
            'shade': False,
            'vmin': -max_val,
            'vmax': max_val,
            'norm': colors.SymLogNorm(
                linthresh=max_val/1000, 
                linscale=max_val/10,
                vmin=-max_val, vmax=max_val, base=10),
        }
        
        
        
        plot = [ax.plot_surface(
            *self.spatial_mesh,
            self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['func'](psi).reshape(self.spatial_mesh[0].shape),
            cmap=cmap,
            
            **self.surface_props
        )]
        if self.simulation_type == 'stationnary':
            V = self.hdf5_file["potential"][:].reshape(self.spatial_mesh[0].shape)
            V_norm = V/np.max(np.abs(V))
            ax.plot_surface(
                *self.spatial_mesh,
                V_norm,
                color='black',
                alpha=0.2,
                linewidth=0,
                antialiased=True
            )
        
        ax.set(
            zlim = (self.surface_props['vmin'], self.surface_props['vmax']), 
            xlabel = r"$x$", 
            ylabel = r"$y$",
            zlabel = f"${self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['label']}$"
            )
        return plot
    
    def _update_1d(self, frame, lines, ax, prop_to_display):
        """Update function for 1D animation."""
        psi = self._get_wave_function(frame)
        lines[0].set_data(self.spatial_mesh, 
                         self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['func'](psi))
        ax.set_title(f"$t$ = {self.temporal_mesh[frame]:.2f}")
        return lines
    
    def _update_2d(self, frame, plot, ax, prop_to_display, cmap):
        """Update function for 2D animation."""
        psi = self._get_wave_function(frame)
        plot[0].remove()
        plot[0] = ax.plot_surface(
            *self.spatial_mesh,
            self.WAVE_FUNCTION_PROPERTIES[prop_to_display]['func'](psi).reshape(self.spatial_mesh[0].shape),
            cmap=cmap,
            **self.surface_props
        )
        ax.set_title(f"$t$ = {self.temporal_mesh[frame]:.2f}")
        return plot

    def animate_solution(self, prop_to_display: str = 'probability_density', 
                        cmap: str = 'cmr.lavender', 
                        save_name: str = None,
                        file_format: str = 'mp4'
                        ):
        """Animate the wave function evolution."""
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d' if self.spatial_mesh.ndim == 3 else None)

        if self.spatial_mesh.ndim == 1:
                lines = self._setup_1d_plot(ax, prop_to_display)
                ani = animation.FuncAnimation(
                    fig=fig,
                    func=lambda frame: self._update_1d(frame, lines, ax, prop_to_display),
                    frames=self.num_iterations,
                    interval=1000*self.dt
                )
        else:
            plot = self._setup_2d_plot(ax, prop_to_display, cmap)
            ani = animation.FuncAnimation(
                fig=fig,
                func=lambda frame: self._update_2d(frame, plot, ax, prop_to_display, cmap),
                frames=self.num_iterations,
                # interval=1000*self.dt,
                interval=1000*self.dt
                )
            
        plt.tight_layout()
        if save_name:
            ani.save(f"{save_name}.{file_format}", **self.animation_props)
        
        # plt.tight_layout()
        plt.show()

    def close(self):
        """Close the HDF5 file."""
        self.hdf5_file.close()
        
    def __del__(self):
        """Ensure the file is closed when the object is deleted."""
        # self.hdf5_file.close()
        if hasattr(self, 'hdf5_file'):
            self.hdf5_file.close()
        # self.close()
        

if __name__ == '__main__':
    file_path = "results/wall_2D.h5"
    reader = SimulationReader(file_path)
    
    print(reader.simulation_type)
    # print(reader.hdf5_file["potential"])
    reader.animate_solution(
        # prop_to_display='probability_density',
        # prop_to_display='real',
        # split_view = True
        # save_name = 'test_2D_proba'
        )
    
    reader.close()