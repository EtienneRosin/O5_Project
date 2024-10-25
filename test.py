import numpy as np
import scipy as sp

# Dimensions de la grille
N_x = 80  # Taille de la grille dans la direction x
N_y = 100  # Taille de la grille dans la direction y
L_x = 10  # Longueur du domaine en x
L_y = 15  # Longueur du domaine en y

# Calcul des fréquences pour chaque direction
freq_x = np.fft.fftfreq(N_x, d=L_x / N_x)
freq_y = np.fft.fftfreq(N_y, d=L_y / N_y)

# Grille de fréquences pour le domaine 2D
Freq_X, Freq_Y = np.meshgrid(freq_x, freq_y, indexing='ij')

# Vectoriser les fréquences pour obtenir un tableau 1D de longueur N_x * N_y
Freq_X_flat = Freq_X.ravel()
Freq_Y_flat = Freq_Y.ravel()

# Calcul de K pour chaque paire de fréquences
K_values = 0.5 * (Freq_X_flat**2 + Freq_Y_flat**2)

# Construction de la matrice diagonale K de taille (N_x * N_y, N_x * N_y)
# K_matrix = np.diag(K_values)
K_sparse = sp.sparse.diags(K_values)

print(f"Taille de K_sparse : {K_sparse.shape}")  # Doit être (N_x * N_y, N_x * N_y)
print(f"Taille mémoire de K_sparse : {K_sparse.data.nbytes} octets")

# print(f"Taille de K_matrix : {K_matrix.shape}")  # Doit être (N_x * N_y, N_x * N_y)
# print(f"Taille mémoire de K_matrix : {K_matrix.data.nbytes} octets")

