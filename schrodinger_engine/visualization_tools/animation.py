import scipy as sp
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmasher as cmr

class WaveFunctionAnimation:
    def __init__(self, folder_path: str):
        self.folder_path = folder_path
        self.time_data = []
        self.spatial_mesh = np.load(os.path.join(self.folder_path, "spatial_mesh.npy"))
        self.iteration_count = self.get_iteration_count()
        
        # self.ax_props = dict()
        
        
    def find_min_max_wave_function(self):
        min_val = float('inf')
        max_val = float('-inf')
        # Parcourt tous les fichiers de fonctions d'onde
        for filename in os.listdir(folder_path):
            if filename.startswith("wave_function_") and filename.endswith(".npy"):
                psi = np.load(os.path.join(folder_path, filename))
                real_part = psi.real
                min_val = min(min_val, real_part.min())
                max_val = max(max_val, real_part.max())
        return min_val, max_val

    def get_iteration_count(self):
        # Compte le nombre de fichiers de fonctions d'onde dans le dossier
        return len([name for name in os.listdir(self.folder_path) if name.startswith("wave_function_")])

    def load_data(self, iteration):
        # Charge les données pour une itération donnée
        psi = np.load(os.path.join(self.folder_path, f"wave_function_{iteration}.npy"))
        time = np.load(os.path.join(self.folder_path, f"time_{iteration}.npy"))
        return psi, time

    def animate_1d(self):
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label='Wave Function')
        ax.set_xlim(self.spatial_mesh.min(), self.spatial_mesh.max())
        ax.set_ylim(-1, 1)  # À ajuster selon les valeurs
        ax.set_xlabel('Space')
        ax.set_ylabel('Wave Function')
        ax.set_title('1D Wave Function Animation')
        ax.grid()
        ax.legend()

        def init():
            line.set_data([], [])
            return line,

        def update(frame):
            psi, time = self.load_data(frame)
            line.set_data(self.spatial_mesh, psi)
            ax.set_title(f't = {time:.3f}')
            return line,

        ani = FuncAnimation(fig, update, frames=self.iteration_count, init_func=init, blit=True, interval=50)
        plt.show()

    def animate_2d(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        cmap = 'cmr.lavender'
        # Initialize surf and timer to None
        surf = None
        timer = None
        plt.tight_layout()
        
        ax.set(
            xlim = (self.spatial_mesh[0].min(), self.spatial_mesh[0].max()),
            ylim = (self.spatial_mesh[1].min(), self.spatial_mesh[1].max()),
            zlim = self.find_min_max_wave_function(),
            xlabel = r"$x$",
            ylabel = r"$y$"
            )

        def init():
            nonlocal surf, timer
            psi, time = self.load_data(0)
            if surf is not None:
                surf.remove()
            psi = psi.reshape(self.spatial_mesh[0].shape)
            field = np.abs(psi)**2
            surf = ax.plot_surface(*self.spatial_mesh, field, cmap = cmap)
            if timer is not None:
                timer.remove()
            timer = ax.text2D(0.05, 0.95, f't = {time:.3f}', transform=ax.transAxes)
            return surf, timer,

        def update(frame):
            nonlocal surf, timer
            psi, time = self.load_data(frame)
            if surf is not None:
                surf.remove()
            psi = psi.reshape(self.spatial_mesh[0].shape)
            field = np.abs(psi)**2
            surf = ax.plot_surface(*self.spatial_mesh, field, cmap = cmap)
            if timer is not None:
                timer.remove()
            timer = ax.text2D(0.05, 0.95, f't = {time:.3f}', transform=ax.transAxes)
            return surf, timer,

        ani = FuncAnimation(fig, update, frames=self.iteration_count, init_func=init, blit=False, interval=50)
        # ani.save("simu.mp4", fps=60, dpi=300)
        plt.show()

    def run_animation(self):
        # Vérifier la dimension des données
        if self.spatial_mesh.ndim == 1:
            self.animate_1d()
        elif self.spatial_mesh.ndim == 3:
            self.animate_2d()
        else:
            raise ValueError("Données de dimension non prise en charge.")


def find_min_max_wave_function(folder_path: str):
    min_val = float('inf')
    max_val = float('-inf')
    # Parcourt tous les fichiers de fonctions d'onde
    for filename in os.listdir(folder_path):
        if filename.startswith("wave_function_") and filename.endswith(".npy"):
            psi = np.load(os.path.join(folder_path, filename))
            real_part = psi.real
            min_val = min(min_val, real_part.min())
            max_val = max(max_val, real_part.max())
    return min_val, max_val


if __name__ == '__main__':
    
    # # Exemple d'utilisation
    # folder_path = "results"  # Spécifie le chemin de ton dossier
    # min_val, max_val = find_min_max_wave_function(folder_path)
    # print(f"Valeur minimale: {min_val}")
    # print(f"Valeur maximale: {max_val}")
    folder_path = "results"  # Spécifie le chemin de ton dossier
    animation = WaveFunctionAnimation(folder_path)
    animation.run_animation()
