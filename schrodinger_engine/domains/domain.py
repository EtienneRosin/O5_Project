import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class Domain:
    def __init__(self, boundaries, N) -> None:
        if len(boundaries) != 2 or boundaries[0] >= boundaries[1]:
            raise ValueError("Invalid boundaries. Ensure boundaries are in the form [min, max] with min < max.")
        if N <= 0:
            raise ValueError("N must be a positive integer.")

        self.boundaries = boundaries
        self.N = N
        
        self.width = np.diff(boundaries)[0]
        self.step = self.width / N

        # self.mesh will be set in the derived classes (SpatialDomain, TemporalDomain)
        self.mesh = None
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}\n| Interval: {self.boundaries} | Points: {self.N}"
    
    def is_centered(self) -> bool:
        """
        Check if the domain is centered around 0.
        """
        # center = (self.boundaries[0] + self.boundaries[1]) / 2
        center = np.mean(self.boundaries)
        return np.abs(center) < 1e-10  # Tolerance for floating-point precision

    def display(self, ax: plt.axes = None, label_x='x', label_y='y') -> None:
        if ax is None:
            fig, ax = plt.subplots()
            fig.canvas.manager.set_window_title(self.__class__.__name__)
        
        # Draw the domain as a blue rectangle on the x-axis
        rect_height = 0.05
        rect = mpatches.Rectangle(
            xy=(self.boundaries[0], -rect_height / 2),
            width=self.width,
            height=rect_height,
            color="blue", alpha=0.1
        )
        ax.add_patch(rect)

        # Plot the mesh points
        if self.mesh is not None:
            ax.scatter(self.mesh, np.zeros_like(self.mesh), c="blue", s=20, zorder=3)
            ax.text(x=self.mesh[0], y=-0.02, s=f"${label_x}_0$", va="top", ha="right", fontsize=10)
            ax.text(x=self.mesh[-1], y=-0.02, s=f"${label_x}_{{{self.N - 1}}}$", va="top", ha="left", fontsize=10)

        ax.set(ylim=(-5 * rect_height, 5 * rect_height))
        ax.set_xlabel(f"{label_x}-axis")
        ax.set_ylabel(f"{label_y}-axis")
        ax.grid(True)





class SpatialDomain(Domain):
    def __init__(self, boundaries, N):
        super().__init__(boundaries, N)
        self.mesh = np.linspace(start=boundaries[0], stop=boundaries[1], num=N, endpoint=False)
    
    def display(self, ax: plt.axes = None) -> None:
        super().display(ax, label_x='x', label_y='y')


class TemporalDomain(Domain):
    def __init__(self, t_end, N, t_init = 0):
        boundaries = [t_init, t_end]
        super().__init__(boundaries, N)
        self.mesh = np.linspace(start=boundaries[0], stop=boundaries[1], num=N, endpoint=True)

    def display(self, ax: plt.axes = None) -> None:
        super().display(ax, label_x='t', label_y='Time')


# Exemple d'utilisation
if __name__ == "__main__":
    spatial_domain = SpatialDomain(boundaries=[-5, 5], N=20)
    temporal_domain = TemporalDomain(t_init = 0, t_end = 10, N=100)

    # Affichage des domaines
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
    
    spatial_domain.display(ax=ax1)
    temporal_domain.display(ax=ax2)

    plt.tight_layout()
    plt.show()