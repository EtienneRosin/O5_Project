import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

x = np.mgrid[:5, :5][0]
data = sp.fft.fft2(x)
# print(sp.fft.fft2(x))

FreqCompRows = np.fft.fftfreq(data.shape[0],d=2)
FreqCompCols = np.fft.fftfreq(data.shape[1],d=2)


class FourierBasis2D:
    """
    Represents a Fourier basis of a 2D domain [0, L_x]x[0, L_y] discretized with steps size dx and dy.
    """
    def __init__(self, L_x: float, L_y: float, dx: float, dy: float) -> None:
        self.dx = dx
        self.dy = dy
        if L_x <= 0:
            raise ValueError(r"L_x should be strictly positive.")
        self.L_x = L_x
        
        if L_y <= 0:
            raise ValueError(r"L_y should be strictly positive.")
        self.L_y = L_y
        
        self.N_x = np.ceil(L_x/dx).astype(int)
        if self.N_x % 2:
            raise ValueError(r"N_x should be even to conserve the parity of the \tilde{c}_{n,m}.")
        self.N_y = np.ceil(L_y/dy).astype(int)
        if self.N_y % 2:
            raise ValueError(r"N_y should be even to conserve the parity of the \tilde{c}_{n,m}.")
        
        self.normalization_factor = 1/np.sqrt(self.L_x * self.L_y)
        
    def k_n_m(self, n: int|np.ndarray, m: int|np.ndarray) -> np.ndarray:
        """
        @brief Returns the wave vector associated to the (n,m)-th function.
        @param n: x-index of the function.
        @param m: y-index of the function.
        @return: wave vector.
        """
        return 2 * np.pi * np.array([n/self.L_x, m/self.L_y])
    
    def k_values(self) -> np.ndarray:
        """
        @brief Returns all the wave vectors associated to the considered domain.
        @return: the wave vectors.
        """
        freq_x = 2 * np.pi * sp.fft.fftfreq(n = self.N_x, d = self.L_x/self.N_x)
        freq_y = 2 * np.pi * sp.fft.fftfreq(n = self.N_y, d = self.L_y/self.N_y)
        return np.array(np.meshgrid(freq_x, freq_y))
    
    def e_n_m(self, n: int, m: int, M: float|np.ndarray) -> float|np.ndarray:
        """
        @brief Evalute the (n, m)-th basis function at (x, y)
        @param n: x-index of the function.
        @param m: y-index of the function.
        @param M: position(s) onto which evaluate the function.
        @return: e_{n,m}(x, y).
        """
        # return 1 / self.L * np.exp(1j * np.dot(self.k_n_m(n, m), M))
        k_nm = self.k_n_m(n, m)  # Taille (2,)

        # Cas où M est une seule position (x, y)
        if isinstance(M, tuple) or (isinstance(M, np.ndarray) and M.ndim == 1 and M.shape[0] == 2):
            # M est un point unique (x, y), donc on fait simplement le produit scalaire
            dot_product = np.dot(k_nm, M)  # Produit scalaire entre k_nm et M (taille 2,)
            return 1 /np.sqrt(self.L_x * self.L_y) * np.exp(1j * dot_product)

        elif isinstance(M, np.ndarray) and M.ndim == 3 and M.shape[0] == 2:
            dot_product = np.tensordot(k_nm, M, axes=(0, 0))
            return 1 /np.sqrt(self.L_x * self.L_y) * np.exp(1j * dot_product)

        else:
            raise ValueError("M doit être un tuple (x, y) ou un tableau de forme (2, N, N).")
    
    def display_e_n_m(self, n_vals: int|np.ndarray[int], m_vals: int|np.ndarray[int]) -> None:
        """
        @brief Display the basis function(s).
        @param n_vals: x-indices of the functions to display.
        @param m_vals: y-indices of the functions to display.
        """

        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"Modes")

        mosaic = [["real"], ["imag"]]
        
        axs = fig.subplot_mosaic(mosaic=mosaic, sharex=True, per_subplot_kw={("real", "imag"): {"projection": "3d"}})

        lst_x = np.arange(start = -self.L_x/2, stop = self.L_x/2, step = self.dx)
        lst_y = np.arange(start = -self.L_y/2, stop = self.L_y/2, step = self.dy)
        domain = np.meshgrid(lst_x, lst_y)
        domain = np.array(domain)
        
        n_vals = np.array([n_vals]).flatten()
        m_vals = np.array([m_vals]).flatten()
        for n,m in zip(n_vals, m_vals):
            e_n_m = self.e_n_m(n,m, domain)
            label = r"e_{" + f"{n},{m}" + r"}"
            axs["real"].plot_surface(*domain, e_n_m.real, label=f"${label}$")
            axs["imag"].plot_surface(*domain, e_n_m.imag)

        fig.legend()
        real_label = r"\Re(e_{" + f"n,m" + r"})"
        axs["real"].set(xlabel=r"$x$", ylabel=r"$y$", zlabel=f"${real_label}$")
        axs["real"].grid()
        imag_label = r"\Im(e_{" + f"n,m" + r"})"
        axs["imag"].set(xlabel=r"$x$", ylabel=r"$y$", zlabel=f"${imag_label}$")
        axs["imag"].grid()

        plt.show()
        
        
if __name__ == '__main__':
    # n = 110
    # N = n
    L_x = 1
    L_y = 1
    
    dx = 0.01
    dy = 0.01
    fb = FourierBasis2D(L_x=L_x, L_y=L_y, dx=dx, dy=dy)
    # fb.display_e_n_m(1,1)
    fb.display_e_n_m([1, 1, 0],[1, 2, 0])
    
    # L = 2
    
    # fb = FourierBasis2D(L = L, N = N)
    # # fb.display_e_n_m(1,1)
    # # fb.display_e_n_m([1, 1, 0],[1, 2, 0])
    # print(fb.k_values().shape)
    # L = 1
    # dx = 0.03
    # print(f"{np.ceil(L/dx).astype(int) = }")
    # lst_x = np.arange(start = -L/2, stop = L/2, step = dx)
    # print(f"{lst_x.shape = }")
    