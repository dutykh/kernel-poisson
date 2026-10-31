# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Approximate leave-one-out (one Newton step from the full fit) for kernel count regression.

At the fit, with g = l'(y, eta), w = l''(y, eta), A = W^{1/2} K W^{1/2} + lam I and
s = W^{1/2} 1, the leverage of observation i (diagonal of the local smoother, intercept
included) is

    h_i = ell_i + lam u_i^2 / (s'u),   ell_i = [M (M + lam I)^{-1}]_ii,   u = A^{-1} s,

that is, the Fisher-weighted ridge leverage score plus the intercept's share. The one-step
leave-one-out natural parameter is

    eta~_i = eta_i + (h_i / (1 - h_i)) g_i / w_i.

Divisions by small weights are avoided: h_i / w_i = G_ii + lam v_i^2 / (w'v), where
G = (W + lam K^{-1})^{-1} = (K - K W^{1/2} A^{-1} W^{1/2} K)/lam (the Laplace posterior
covariance of f at the data, times lam) and v = (K W + lam I)^{-1} 1 = (1 - K W^{1/2} u)/lam.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def leverages(K, w, lam):
    """Return (h, h / w) by one Cholesky factorisation of A (O(N^3))."""
    sw = np.sqrt(w)
    A = sw[:, None] * K * sw[None, :]
    A[np.diag_indices_from(A)] += lam
    L = sla.cholesky(A, lower=True, check_finite=False)
    B = sla.solve_triangular(L, sw[:, None] * K, lower=True, check_finite=False)  # L^{-1} W^{1/2} K
    G_diag = (np.diag(K) - np.einsum("ij,ij->j", B, B)) / lam
    u = sla.cho_solve((L, True), sw, check_finite=False)
    v = (1.0 - K @ (sw * u)) / lam
    hw = G_diag + lam * v * v / float(w @ v)
    return w * hw, hw


def alo_eta(eta, g, w, h, hw):
    """One-step leave-one-out natural parameters."""
    return eta + hw * g / (1.0 - h)


def cook_distances(y, mu, h, w):
    """Cook-type distances c_i = (Pearson residual)^2 h_i / (1 - h_i) for the Poisson loss."""
    return (y - mu) ** 2 / np.maximum(mu, 1e-300) * h / (1.0 - h)
