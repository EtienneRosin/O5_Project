from schrodinger_engine.utils.domain import Domain
import numpy as np
import scipy.fft as spfft
import scipy as sp
import matplotlib.pyplot as plt

class FourierBasis:
    """
    Represents a Fourier basis associated with a domain of dimension 1 or 2.
    """
    def __init__(self, domain: Domain) -> None:
        """
        Initialize the Fourier basis with a domain.
        
        Parameters:
        - domain: Domain object representing the spatial or temporal domain.
        """
        self.domain = domain
        self.frequencies = self.get_frequencies()
    
    def get_frequencies(self):
        """
        Calculate the frequencies based on the domain dimensions.
        """
        factor = 2 * np.pi
        if self.domain.dim == 1:
            return factor * np.fft.fftfreq(self.domain.N[0], d=self.domain.step[0])
        elif self.domain.dim == 2:
            freq_x = factor * np.fft.fftfreq(self.domain.N[0], d=self.domain.step[0])
            freq_y = factor * np.fft.fftfreq(self.domain.N[1], d=self.domain.step[1])
            return np.meshgrid(freq_x, freq_y, indexing='ij')
        else:
            raise ValueError("Only 1D and 2D domains are currently supported.")
    
    def fft(self, f):
        """
        Apply the forward FFT on the function f.
        
        Parameters:
        - f: array-like, the input function values to transform.
        
        Returns:
        - Transformed values in frequency space.
        """
        return spfft.fftn(f, norm="forward")
    
    def ifft(self, F):
        """
        Apply the inverse FFT on the function F in frequency space.
        
        Parameters:
        - F: array-like, the input frequency-space values to transform.
        
        Returns:
        - Transformed values in real space.
        """
        return spfft.ifftn(F, norm="forward")
    
    
    # def get_second_derivative(self, f):
    #     k_vals = self.get_frequencies()
    #     return self.ifft(((1j * k_vals)**2)*self.fft(f))
    def get_second_derivative(self, f):
        """
        Calculate the second derivative of a function f using Fourier transforms.
        
        Parameters:
        - f: array-like, the input function values to transform.

        Returns:
        - The second derivative of f in real space.
        """
        k_vals = self.get_frequencies()
        F = self.fft(f)

        if self.domain.dim == 1:
            kx = k_vals
            second_derivative_freq = (1j * kx) ** 2
        elif self.domain.dim == 2:
            kx, ky = k_vals
            second_derivative_freq = (1j * kx) ** 2 + (1j * ky) ** 2
        else:
            raise ValueError("Only 1D and 2D domains are currently supported.")

        return self.ifft(second_derivative_freq * F)

    def plot_spectrum(self, f):
        """
        Display the amplitude spectrum of a function f in the frequency domain.
        
        Parameters:
        - f: array-like, the function values to transform and plot in the frequency domain.
        """
        F = self.fft(f)
        amplitude_spectrum = np.abs(F)
        
        if self.domain.dim == 1:
            plt.figure()
            plt.stem(sp.fft.fftshift(self.frequencies), sp.fft.fftshift(amplitude_spectrum), 'r-')
            plt.xlabel("Frequency (Hz)")
            plt.ylabel("Amplitude")
            plt.title("Amplitude Spectrum in 1D")
            plt.grid()
            plt.show()
        
        elif self.domain.dim == 2:
            plt.figure()
            plt.imshow(amplitude_spectrum, extent=(self.frequencies[0].min(), self.frequencies[0].max(),
                                                   self.frequencies[1].min(), self.frequencies[1].max()),
                       origin='lower', aspect='auto', cmap="plasma")
            plt.xlabel("Frequency X (Hz)")
            plt.ylabel("Frequency Y (Hz)")
            plt.title("Amplitude Spectrum in 2D")
            plt.colorbar(label="Amplitude")
            plt.show()
        else:
            raise ValueError("Only 1D and 2D domains are supported for plotting spectrum.")
        
if __name__ == '__main__':
    from schrodinger_engine.utils.domain import Domain

    domain_1d = Domain(boundaries=[(-5, 5)], N=200)
    basis_1d = FourierBasis(domain_1d)

    def f1d(x):
        return np.sin(2 * np.pi * x)
    
    def f1d_second(x):
        return -f1d(x)*(2*np.pi)**2

    f1d_values = f1d(domain_1d.get_mesh())
    f1d_second_derivative = basis_1d.get_second_derivative(f1d_values)
    
    # Afficher les résultats 1D
    fig, ax = plt.subplots()
    ax.plot(domain_1d.get_mesh(), f1d_second(domain_1d.get_mesh()) - f1d_second_derivative.real, label='Dérivée seconde de f en 1D')
    ax.legend()
    plt.title("Dérivée seconde en 1D")
    plt.xlabel("x")
    plt.ylabel("Dérivée seconde")
    plt.grid()
    plt.show()

    # Exemple 2D
    domain_2d = Domain(boundaries=[(-10, 10), (-10, 10)], N=[100, 100])
    basis_2d = FourierBasis(domain_2d)
    X, Y = domain_2d.get_mesh()

    def f2d(x, y):
        return np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    
    def f2d_second(x, y):
        return -f2d(x, y)*2*(2*np.pi)**2
    

    f2d_values = f2d(*domain_2d.get_mesh())
    f2d_second_derivative = basis_2d.get_second_derivative(f2d_values)
    
    
    # Afficher les résultats 2D
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(f2d_values, extent=(-10, 10, -10, 10), origin='lower', aspect='auto', cmap='plasma')
    ax[0].set_title("Fonction d'origine en 2D")
    # ax[0].colorbar(label='Valeur de f')

    im = ax[1].imshow(f2d_second(*domain_2d.get_mesh()) - f2d_second_derivative.real, extent=(-10, 10, -10, 10), origin='lower', aspect='auto', cmap='plasma')
    ax[1].set_title("Dérivée seconde de f en 2D")
    # ax[1].colorbar(label='Dérivée seconde')
    fig.colorbar(im, ax = ax[1])
    
    plt.show()