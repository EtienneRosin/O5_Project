import numpy as np
import matplotlib.pyplot as plt

from schrodinger_engine.utils.domain import Domain

class PlaneWave:
    r"""
    Represent a plane wave of form : e^{-i \vec{k} \cdot \vec{x}}.
    """
    def __init__(self, k: float | list | np.ndarray, dim: int) -> None:
        """
        Initialize the PlaneWave object.
        
        Parameters
        ----------
        k: float | list | np.ndarray
            Wave number/vector
        dim: int
            Dimension of the domain.
        """
        self.dim = dim
        self.k = self._validate_k(k)

    def _validate_k(self, k: float | list | np.ndarray) -> np.ndarray:
        """
        Validate the k input.

        Parameters
        ----------
        k: float | list | np.ndarray
            Wave number/vector

        Returns
        -------
        np.ndarray
            Validated wave vector
        """
        k = np.asarray(k)
        if k.shape != (self.dim,):
            raise ValueError(f"k shape should be ({self.dim},) as it is a {self.dim}D vector.")
        return k

    def __call__(self, x: list | np.ndarray) -> np.ndarray:
        """
        Evaluate the plane wave at the given points.

        Parameters
        ----------
        x: list | np.ndarray
            Points where the wave is evaluated. Should be of shape (dim, N), where N is the number of points.

        Returns
        -------
        np.ndarray
            Values of the plane wave at x.
        """
        x = np.asarray(x)
        if x.ndim not in [1, 2]:
            raise ValueError(f"Input points should be of shape ({self.dim}, N) for a {self.dim}D wave.")
        elif x.ndim == 1:
            return np.exp(-1j * self.k * x)
        else :
            return np.exp(-1j * np.dot(self.k, x))

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    k_1d = [2]
    plane_wave_1d = PlaneWave(k_1d, dim=1)
    # x_1d = np.linspace(-5, 5, 100).reshape(1, -1)
    x_1d = np.linspace(-5, 5, 100)
    print(f"{x_1d.ndim = }")
    wave_values_1d = plane_wave_1d(x_1d)
    plt.figure()
    plt.plot(x_1d.ravel(), wave_values_1d.real)
    plt.xlabel('x')
    plt.ylabel('Real Part of Plane Wave')
    plt.title('1D Plane Wave')
    plt.show()

    # Define wave vector for 2D
    k_2d = [2, 3]
    plane_wave_2d = PlaneWave(k_2d, dim=2)
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    points = np.vstack([X.ravel(), Y.ravel()])
    
    wave_values_2d = plane_wave_2d(points).reshape(X.shape)

    plt.figure()
    plt.contourf(X, Y, wave_values_2d.real)
    plt.colorbar(label='Real Part of Plane Wave')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Plane Wave')
    plt.show()
