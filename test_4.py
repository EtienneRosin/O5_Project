import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

# Créer un signal d'essai simple
L = 10  # Longueur du domaine
N = 200  # Nombre de points
x = np.linspace(-L/2, L/2, N, endpoint=False)
signal = 3 * np.sin(2 * np.pi * 1 * x)  # Amplitude de 3 à 1 Hz

# Calculer la FFT
F = 2 * sp.fft.fft(signal)
amplitude_spectrum = np.abs(F)
reconstructed_signal = np.fft.ifft(F)

# Vérifier la somme des amplitudes
print("Max amplitude in original signal:", np.max(signal))
print("Max amplitude in FFT (scaled):", np.max(amplitude_spectrum) / N)

# Plotter les résultats
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(x, signal)
plt.plot(x, reconstructed_signal)
plt.title("Signal original")
plt.subplot(2, 1, 2)
plt.stem(np.fft.fftfreq(N, d=(L/N)), amplitude_spectrum)
plt.title("Spectre d'amplitude")
plt.show()