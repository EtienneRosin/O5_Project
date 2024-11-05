from schrodinger_engine.initial_conditions import InitialCondition

import numpy as np

class GaussianWavePacket(InitialCondition):
    """
    Represent a Gaussian wave packet.
    
    Notes
    -----
    The initial energy of such a wave packet is: :math:`E = \frac{\hbar^2 \vec{k}_0^2}{2m}`
    
    In the case of the free propagation of such a wave packet, we have :
        - phase velocity : :math:`\vec{v}_g = \frac{\hbar \vec{k}_0}{2m}`
        - group velocity : :math:`\vec{v}_g = \frac{\hbar \vec{k}_0}{m}`
        - dispersion relation: :math:`\omega(\vec{k}) = \frac{\hbar \vec{k}^2}{2m}`
        
    References
    ----------
    .. [1] C. Cohen-Tannoudji, B. Diu, F. Laloë, MÉCANIQUE QUANTIQUE - TOME 1, Nouvelle édition
    """
    def __init__(self, x_0: float | list | np.ndarray, k_0: float | list | np.ndarray, sigma: float, dim: int) -> None:
        """
        Initialize the GaussianWavePacket object.
        
        Parameters
        ----------
        x_0: float | list | np.ndarray
            Initial position.
        k_0: float | list | np.ndarray
            Initial wave vector.
        sigma: float
            Width of the Gaussian packet.
        dim: int
            Dimension of the domain.
        """
        self.dim = dim
        self.x_0 = self._validate_vector(x_0, dim)
        self.k_0 = self._validate_vector(k_0, dim)
        self.sigma = sigma
        
        self.wave_number = np.linalg.norm(self.k_0)
        
        self.omega = (self.wave_number**2)/2            # temporal angular frequency in the free case
        
        self.phase_velocity = self.wave_number/2        # phase velocity in the free case
        self.group_velocity = 2 * self.phase_velocity   # group velocity in the free case
        self.initial_energy = (self.wave_number**2)/2   # initial energy in the free case

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
        envelope = np.exp(-np.sum((x - self.x_0[:, np.newaxis])**2, axis=0) / (4 * self.sigma**2))

        # Plane wave part
        plane_wave = np.exp(1j * np.dot(self.k_0, x))
        psi = envelope * plane_wave
        if x.size > 1:
            psi /= np.linalg.norm(psi)
        
        return psi

# Example usage
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ## 1D Gaussian Wave Packet
    x_0_1d = [0]
    k_0_1d = [3]
    sigma_1d = 1
    g_wave_packet_1d = GaussianWavePacket(x_0=x_0_1d, k_0=k_0_1d, sigma=sigma_1d, dim=1)
    g_wave_packet_1d.display()
    plt.show()
    
    
    # x_1d = np.linspace(-10, 10, 500).reshape(1, -1)
    # x_1d = np.linspace(-10, 10, 500)
    # print(f"{x_1d.size = }")
    # wave_values_1d = g_wave_packet_1d(x_1d)
    
    # plt.figure()
    # plt.plot(x_1d.ravel(), wave_values_1d.real, label='Real Part')
    # plt.plot(x_1d.ravel(), wave_values_1d.imag, label='Imaginary Part')
    # plt.xlabel('x')
    # plt.ylabel('Wave Packet')
    # plt.title('1D Gaussian Wave Packet')
    # plt.legend()
    # plt.show()
    
    # # 2D Gaussian Wave Packet
    x_0_2d = [0, 0]
    k_0_2d = [0.5, 0.25]
    sigma_2d = 10
    g_wave_packet_2d = GaussianWavePacket(x_0=x_0_2d, k_0=k_0_2d, sigma=sigma_2d, dim=2)
    g_wave_packet_2d.display()
    plt.show()
    
    # x = np.linspace(-10, 10, 100)
    # y = np.linspace(-10, 10, 100)
    # X, Y = np.meshgrid(x, y)
    # points_2d = np.vstack([X.ravel(), Y.ravel()])
    # print(f"{points_2d.size = }")
    # wave_values_2d = g_wave_packet_2d(points_2d).reshape(X.shape)
    
    
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # # plt.figure()
    # # ax.contourf(X, Y, wave_values_2d.real)
    # ax.plot_surface(X,Y, wave_values_2d.real)
    # # plt.colorbar(label='Real Part of Gaussian Wave Packet')
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # # plt.title('2D Gaussian Wave Packet')
    # plt.show()
