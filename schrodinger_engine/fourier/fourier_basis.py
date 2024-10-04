"""
@file fourier_basis.py
@brief Implementation of a FourierBasis class that handle operation of a Fourier basis defined on a domain [-L/2, L/2].
@author Etienne Rosin 
@version 0.1
@date 15/09/2024
"""
from typing import Union

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt


class FourierBasis:
    """
    @class FourierBasis
    @brief Provides Fourier basis functions on [-L/2, L/2] which is dicretized as x_n = -L/2 + nL/N_x, ∀n = 0, ..., N_x - 1.
    """

    def __init__(self, L: float, N_x: int) -> None:
        """
        @brief Initialize the basis
        @param L: Length of the symetric domain of definition of the basis functions.
        @param N_x: number of points of the dicretization.
        """
        if N_x % 2:
            raise ValueError(
                r"N_x should be even to conserve the parity of the \tilde{c}_k.")
        # @var L
        # Length of the symetric domain of definition of the basis functions.
        self.L: float = L
        # @var N_x
        # number of points of the dicretization.
        self.N_x: int = N_x

        # @var mesh
        # discretized domain.
        self.mesh: np.ndarray[float] = np.linspace(
            start=-L/2, stop=L / 2, num=N_x, endpoint=False)

    def k_n(self, n: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """
        @brief Returns the "wavelength" associated to the n-th function.
        @param n: index of the function.
        @return: the "wavelength".
        """
        return 2 * np.pi * n / self.L

    def k_values(self):
        return sp.fft.fftfreq(n=self.N_x, d=self.L/self.N_x) * 2 * np.pi

    def e_n(self, n: int, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        @brief Evalute the n-th function at x
        @param n: index of the function.
        @param x: position(s) onto which evaluate the function.
        @return: e_n(x).
        """
        return 1 / np.sqrt(self.L) * np.exp(1j * self.k_n(n) * x)

    def evaluate_basis(self, n_vals: Union[list[int], np.ndarray[int]], x_vals: Union[float, np.ndarray[float]]) -> np.ndarray:
        """
        @brief Return an array with values of each combinaison of x in x_vals and n in n_vals.
        @param n_vals: Indices of the functions to evaluate.
        @param x_vals: Points where to evaluate the functions
        """
        n_vals = np.array(n_vals)
        x_vals = np.array(x_vals)

        return 1 / np.sqrt(self.L) * np.exp(1j * self.k_n(n_vals)[:, None] * x_vals)

    def display_e_n(self, n_vals: Union[int, np.ndarray[int]]) -> None:
        """
        @brief Display the basis function(s).
        @param n_vals: Indices of the functions to display.
        """

        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"Modes")

        mosaic = [["real"], ["imag"]]
        axs = fig.subplot_mosaic(mosaic=mosaic, sharex=True)

        domain = self.mesh
        n_vals = np.array([n_vals]).flatten()
        for n in n_vals:
            e_n = self.e_n(n, domain)
            # real_label = r"\Re(e_{" + f"{i}" + r"})"
            # imag_label = r"\Im(e_{" + f"{i}" + r"})"
            label = r"e_{" + f"{n}" + r"}"
            axs["real"].plot(domain, e_n.real, label=f"${label}$")
            axs["imag"].plot(domain, e_n.imag)

        fig.legend()
        real_label = r"\Re(e_{" + f"n" + r"})"
        axs["real"].set(ylabel=f"${real_label}$")
        axs["real"].grid()
        imag_label = r"\Im(e_{" + f"n" + r"})"
        axs["imag"].set(xlabel=r"$x$", ylabel=f"${imag_label}$")
        axs["imag"].grid()

        plt.show()

    def project_function(self, f: Union[callable, np.ndarray[float]]) -> np.ndarray[complex]:
        """
        @brief Project the function onto the Fourier basis
        @param f: The considered function or the values of a function on the mesh.
        @return c_tilde: projection coefficients of f.
        """
        if callable(f):
            f_values = f(self.mesh)
        elif isinstance(f, np.ndarray):
            if f.shape != self.mesh.shape:
                raise ValueError(f"Expected array of the shape of the mesh : {
                                 self.mesh.shape}, but got : {f.shape}")
            f_values = f
        else:
            raise ValueError("f should be a callable or a numpy array.")

        c_tilde = (np.sqrt(self.L)/self.N_x)*sp.fft.fft(f_values)
        return c_tilde

    def display_decomposition(self, f: Union[callable, np.ndarray[float]]) -> None:
        """
        @brief Display the decomposition of the function onto the Fourier basis.
        @param f: The considered function or the values of a function on the mesh.
        """
        # Compute Fourier coefficients
        c_tilde = self.project_function(f)
        # k_vals = np.fft.fftfreq(self.N_x, d=self.L / self.N_x)
        k_vals = np.arange(self.N_x)

        fig = plt.figure()
        fig.canvas.manager.set_window_title(f"Fourier decomposition of f")
        mosaic = [["real"], ["imag"], ["magnitude"]]
        axs = fig.subplot_mosaic(mosaic=mosaic, sharex=True)

        marker_props = dict(markersize=2)
        line_props = dict(linewidth=0.75)

        stem = axs["real"].stem(k_vals, c_tilde.real, basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["real"].set(ylabel=r"Real part $\Re(\tilde{c}_k)$")

        stem = axs["imag"].stem(k_vals, c_tilde.imag, basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["imag"].set(ylabel=r"Imaginary part $\Im(\tilde{c}_k)$")
        # axs["imag"].grid()

        stem = axs["magnitude"].stem(k_vals, np.abs(c_tilde), basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["magnitude"].set(ylabel=r"Magnitude $|\tilde{c}_k|$")

        # Add a vertical line to indicate N_x / 2 (Nyquist frequency)
        for ax_name in axs:
            axs[ax_name].axvline(self.N_x / 2, linestyle="--",
                                 color="red", label="$N_x/2$")
            axs[ax_name].legend()
            axs[ax_name].grid()

        axs["magnitude"].set_xlabel("k (Fourier mode)")

        plt.tight_layout()
        plt.show()

    def compare_decomposition(self, f: Union[callable, np.ndarray[float]]) -> None:
        """
        @brief Display the comparaison of decomposition of the function onto the Fourier basis and its real values.
        @param f: The considered function or the values of a function on the mesh.
        """
        f_values = f(self.mesh) if callable(f) else f

        c_tilde = self.project_function(f)

        f_values_tilde = sp.fft.ifft((self.N_x/np.sqrt(self.L)) * c_tilde).real

        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title(
            f"Comparison between the Fourier decomposition of f and f.")
        ax.plot(self.mesh, f_values, label=r"$f$")
        ax.plot(self.mesh, f_values_tilde,
                label=r"$\tilde{f}$", linestyle="--")

        ax.set(xlabel="$x$")
        ax.grid()
        ax.legend()
        plt.show()

    def project_derivative_of_function(self, f: Union[callable, np.ndarray[float]], order: int = 1, return_real_domain: bool = False) -> np.ndarray[complex]:
        """
        @brief Project the order-th derivative of f onto the Fourier basis or return the derivative in the time domain.
        @param f: The considered function or the values of a function on the mesh.
        @param order: order of the derivation.
        @param return_time_domain: If True, return the derivative in the time domain.
        @return: projection coefficients of f^(order) or the derivative in the time domain.
        """
        c_tilde = self.project_function(f)
        k_values = sp.fft.fftfreq(
            n=self.N_x, d=self.L/self.N_x) * 2 * np.pi

        # Eviter la multiplication par zéro pour le terme constant.
        # k_values[0] = 0

        c_tilde *= (1j * k_values)**order

        if return_real_domain:
            return sp.fft.ifft((self.N_x/np.sqrt(self.L))*c_tilde).real
        return c_tilde

    def display_derivative_decomposition(self, f: Union[callable, np.ndarray[float]], order: int = 1) -> None:
        """
        @brief Display the decomposition of the derivative of a function onto the Fourier basis.
        @param f: The considered function or the values of a function on the mesh.
        @param order: The order of the derivative to compare (default is 1 for the first derivative).
        """
        # Compute Fourier coefficients
        c_tilde = self.project_derivative_of_function(f, order=order)
        k_vals = np.arange(self.N_x)

        fig = plt.figure()

        fig.canvas.manager.set_window_title(
            f"Fourier decomposition of f^({order}).")
        mosaic = [["real"], ["imag"], ["magnitude"]]
        axs = fig.subplot_mosaic(mosaic=mosaic, sharex=True)

        marker_props = dict(markersize=2)
        line_props = dict(linewidth=0.75)

        stem = axs["real"].stem(k_vals, c_tilde.real, basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["real"].set(ylabel=r"Real part $\Re(\tilde{c}_k)$")

        stem = axs["imag"].stem(k_vals, c_tilde.imag, basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["imag"].set(ylabel=r"Imaginary part $\Im(\tilde{c}_k)$")
        # axs["imag"].grid()

        stem = axs["magnitude"].stem(k_vals, np.abs(c_tilde), basefmt=" ")
        stem[0].set(**marker_props)
        stem[1].set(**line_props)
        axs["magnitude"].set(ylabel=r"Magnitude $|\tilde{c}_k|$")

        # Add a vertical line to indicate N_x / 2 (Nyquist frequency)
        for ax_name in axs:
            axs[ax_name].axvline(self.N_x / 2, linestyle="--",
                                 color="red", label="$N_x/2$")
            axs[ax_name].legend()
            axs[ax_name].grid()

        axs["magnitude"].set_xlabel("k (Fourier mode)")

        plt.tight_layout()
        plt.show()

    def compare_derivative_decomposition(self, f: Union[callable, np.ndarray[float]],
                                         f_prime: Union[callable, np.ndarray[float]], order: int = 1) -> None:
        """
        @brief Display the comparison of decomposition of the derivative of a function onto the Fourier basis and its real values.
        @param f: The considered function or its values on the mesh.
        @param f_prime: The considered derivative of the function or its values on the mesh.
        @param order: The order of the derivative to compare (default is 1 for the first derivative).
        """
        # Handle f_prime: callable or array
        if callable(f_prime):
            f_prime_values = f_prime(self.mesh)
        elif isinstance(f_prime, np.ndarray):
            if f_prime.shape != self.mesh.shape:
                raise ValueError(f"Expected array of shape {
                                 self.mesh.shape}, but got {f_prime.shape}")
            f_prime_values = f_prime
        else:
            raise ValueError("f_prime should be a callable or a numpy array.")

        # Project the derivative of f in Fourier space and return it in the real domain
        f_prime_tilde_values = self.project_derivative_of_function(
            f, order=order, return_real_domain=True)

        # Plot the comparison
        fig = plt.figure()
        fig.canvas.manager.set_window_title(
            f"Comparison between the Fourier decomposition of f^({order}) and f^({order}).")

        ax = fig.add_subplot()

        ax.plot(self.mesh, f_prime_values,
                label=fr"$\frac{{d^{{{order}}} f}}{{dx^{{{order}}}}}$", color='blue')
        ax.plot(self.mesh, f_prime_tilde_values, label=fr"$\tilde{{\frac{{d^{{{
                order}}} f}}{{dx^{{{order}}}}}}}$", linestyle="--", color='orange')

        ax.set(xlabel="$x$")
        ax.grid()
        ax.legend()
        plt.show()


if __name__ == '__main__':
    pass
