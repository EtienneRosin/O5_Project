import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cmasher as cmr

class WaveFunctionAnimation:
    def __init__(self, folder_path: str, save_animation: bool = False, fps: int = 60, dpi: int = 300):
        self.folder_path = folder_path
        self.save_animation = save_animation
        self.fps = fps
        self.dpi = dpi
        self.time_data = []
        self.spatial_mesh = np.load(os.path.join(self.folder_path, "spatial_mesh.npy"))
        self.iteration_count = self.get_iteration_count()

    def get_iteration_count(self):
        """Compter le nombre de fichiers de fonctions d'onde dans le dossier."""
        return len([name for name in os.listdir(self.folder_path) if name.startswith("wave_function_")])

    def find_min_max_wave_function(self):
        """Trouver les valeurs minimales et maximales des fonctions d'onde."""
        min_val = float('inf')
        max_val = float('-inf')
        for filename in os.listdir(self.folder_path):
            if filename.startswith("wave_function_") and filename.endswith(".npy"):
                psi = np.load(os.path.join(self.folder_path, filename))
                real_part = psi.real
                min_val = min(min_val, real_part.min())
                max_val = max(max_val, real_part.max())
        return min_val, max_val

    def load_data(self, iteration):
        """Charger les données pour une itération donnée."""
        psi = np.load(os.path.join(self.folder_path, f"wave_function_{iteration}.npy"))
        time = np.load(os.path.join(self.folder_path, f"time_{iteration}.npy"))
        return psi, time

    def animate_1d(self):
        """Créer une animation pour la fonction d'onde 1D."""
        fig, ax = plt.subplots()
        line, = ax.plot([], [], label='Wave Function')
        ax.set_xlim(self.spatial_mesh.min(), self.spatial_mesh.max())
        ax.set_ylim(*self.find_min_max_wave_function())  # Dynamique
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
            line.set_data(self.spatial_mesh, np.abs(psi)**2)
            ax.set_title(f't = {time:.3f}')
            return line,

        ani = FuncAnimation(fig, update, frames=self.iteration_count, init_func=init, blit=True, interval=1000/self.fps)
        
        if self.save_animation:
            ani.save("1d_wave_function_animation.mp4", fps=self.fps, dpi=self.dpi)

        plt.show()

    def animate_2d(self):
        """Créer une animation pour la fonction d'onde 2D avec densité de probabilité et phase."""
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        cmap = 'cmr.lavender'
        
        # Définir les limites des axes
        ax.set_xlim(self.spatial_mesh[0].min(), self.spatial_mesh[0].max())
        ax.set_ylim(self.spatial_mesh[1].min(), self.spatial_mesh[1].max())
        ax.set_zlim(*self.find_min_max_wave_function())
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

        # Initialisation des surfaces et du timer
        psi_init, time_init = self.load_data(0)
        prob_density = np.abs(psi_init)**2  # Densité de probabilité
        phase = np.angle(psi_init)  # Phase

        # Créer la surface initiale
        surf = ax.plot_surface(*self.spatial_mesh, prob_density.reshape(self.spatial_mesh[0].shape), cmap=cmap, facecolors=plt.cm.viridis((phase + np.pi) / (2 * np.pi)), rstride=1, cstride=1, antialiased=True)
        timer = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)

        def init():
            nonlocal surf, timer
            psi, time = self.load_data(0)
            prob_density = np.abs(psi)**2  # Densité de probabilité
            phase = np.angle(psi)  # Phase
            surf.remove()
            surf = ax.plot_surface(*self.spatial_mesh, prob_density.reshape(self.spatial_mesh[0].shape), cmap=cmap, facecolors=plt.cm.viridis((phase + np.pi) / (2 * np.pi)), rstride=1, cstride=1, antialiased=True)
            timer.set_text(f't = {time:.3f}')
            return surf, timer

        def update(frame):
            nonlocal surf, timer
            psi, time = self.load_data(frame)
            prob_density = np.abs(psi)**2  # Densité de probabilité
            phase = np.angle(psi)  # Phase
            surf.remove()
            surf = ax.plot_surface(*self.spatial_mesh, prob_density.reshape(self.spatial_mesh[0].shape), cmap=cmap, facecolors=plt.cm.viridis((phase + np.pi) / (2 * np.pi)), rstride=1, cstride=1, antialiased=True)
            timer.set_text(f't = {time:.3f}')
            return surf, timer

        ani = FuncAnimation(fig, update, frames=self.iteration_count, init_func=init, blit=False, interval=1000/self.fps)

        if self.save_animation:
            ani.save("2d_wave_function_animation.mp4", fps=self.fps, dpi=self.dpi)

        plt.show()

    def run_animation(self):
        """Détermine la dimension des données et exécute l'animation appropriée."""
        if self.spatial_mesh.ndim == 1:
            self.animate_1d()
        elif self.spatial_mesh.ndim == 3:
            self.animate_2d()
        else:
            raise ValueError("Données de dimension non prise en charge.")

if __name__ == '__main__':
    folder_path = "results"  # Spécifie le chemin de ton dossier
    save_animation = False  # Change à False si tu ne veux pas sauvegarder
    animation = WaveFunctionAnimation(folder_path, save_animation=save_animation)
    animation.run_animation()