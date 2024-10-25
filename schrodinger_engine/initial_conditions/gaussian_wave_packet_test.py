import numpy as np

class GaussianWavePacket:
    """
    Represent a Gaussian wave packet.
    """
    def __init__(self, x0: float | list | np.ndarray, p0: float | list | np.ndarray, sigma: float, dim: int) -> None:
        """
        Initialize the GaussianWavePacket object.
        
        Parameters
        ----------
        x0: float | list | np.ndarray
            Initial position.
        p0: float | list | np.ndarray
            Initial momentum.
        sigma: float
            Width of the Gaussian packet.
        dim: int
            Dimension of the domain.
        """
        self.dim = dim
        self.x0 = self._validate_vector(x0, dim)
        self.p0 = self._validate_vector(p0, dim)
        self.sigma = sigma

    def _validate_vector(self, vec: float | list | np.ndarray, dim: int) -> np.ndarray:
        """
        Validate the input vector.
        
        Parameters
        ----------
        vec: float | list | np.ndarray
            Vector to validate.
        dim: int
            Dimension of the vector.
        
        Returns
        -------
        np.ndarray
            Validated vector.
        """
        vec = np.asarray(vec)
        if vec.shape == ():
            vec = np.array([vec])
        if vec.shape != (dim,):
            raise ValueError(f"Vector shape should be ({dim},).")
        return vec

    def __call__(self, x: list|np.ndarray) -> np.ndarray:
        """
        Evaluate the Gaussian wave packet at the given points.
        
        Parameters
        ----------
        x: list | np.ndarray
            Points where the wave packet is evaluated. Should be of shape (dim, N), where N is the number of points.
        
        Returns
        -------
        np.ndarray
            Values of the Gaussian wave packet at x.
        """
        x = np.asarray(x)
        
        if x.ndim == 1:
            x = x.reshape(1, -1)
        
        if x.ndim == 3 and x.shape[0] == 2:
            
            x = np.vstack([x[0].ravel(), x[1].ravel()])
        
        if x.ndim != 2 or x.shape[0] != self.dim:
            raise ValueError(f"Input points should be of shape ({self.dim}, N) for a {self.dim}D wave packet.")

        # Gaussian envelope
        envelope = np.exp(-np.sum((x - self.x0[:, np.newaxis])**2, axis=0) / (4 * self.sigma**2))

        # Plane wave part
        plane_wave = np.exp(1j * np.dot(self.p0, x))

        return envelope * plane_wave

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # 1D Gaussian Wave Packet
    x0_1d = [0]
    p0_1d = [2]
    sigma_1d = 1
    g_wave_packet_1d = GaussianWavePacket(x0=x0_1d, p0=p0_1d, sigma=sigma_1d, dim=1)
    
    # x_1d = np.linspace(-10, 10, 500).reshape(1, -1)
    x_1d = np.linspace(-10, 10, 500)
    wave_values_1d = g_wave_packet_1d(x_1d)
    
    plt.figure()
    plt.plot(x_1d.ravel(), wave_values_1d.real, label='Real Part')
    plt.plot(x_1d.ravel(), wave_values_1d.imag, label='Imaginary Part')
    plt.xlabel('x')
    plt.ylabel('Wave Packet')
    plt.title('1D Gaussian Wave Packet')
    plt.legend()
    plt.show()
    
    # 2D Gaussian Wave Packet
    x0_2d = [0, 0]
    p0_2d = [2, 3]
    sigma_2d = 1
    g_wave_packet_2d = GaussianWavePacket(x0=x0_2d, p0=p0_2d, sigma=sigma_2d, dim=2)
    
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)
    points_2d = np.vstack([X.ravel(), Y.ravel()])
    wave_values_2d = g_wave_packet_2d(points_2d).reshape(X.shape)
    
    plt.figure()
    plt.contourf(X, Y, wave_values_2d.real)
    plt.colorbar(label='Real Part of Gaussian Wave Packet')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('2D Gaussian Wave Packet')
    plt.show()
