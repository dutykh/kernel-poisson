# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Pointwise uncertainty of the fitted log-rate in the Gaussian reading of the penalized fit.

At the fit, with Fisher weights w_i = l''(y_i, eta_i) and penalty lam, the Hessian of the objective
on the joint parameter (b, f) is H = sum_i w_i phi(x_i) (x) phi(x_i) + lam P, phi(x) = (1, k_x).
Reading the penalized fit as a posterior mode with prior precision lam P (flat in the intercept),
the variance of the log-rate b + f(x) is

    q(x) = <phi(x), H^{-1} phi(x)>
         = (k(x,x) - s_x' (M + lam I)^{-1} s_x) / lam
           + (1 - u' (M + lam I)^{-1} s_x)^2 / (lam u' (M + lam I)^{-1} u),

with M = W^{1/2} K W^{1/2}, s_x = W^{1/2} k_x and u = W^{1/2} 1 (the kriging variance with an unknown
constant mean). At a data point, q(x_i) = h_i / w_i, the leverage divided by the weight. In the
projected model, q(x) = u_x' A^{-1} u_x with u_x = (1, K_xL) and A the Hessian in (b, beta).
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .kernels import gram


def log_rate_variance(est, X, y, exposure, Xnew):
    """Variance q(x) of the fitted log-rate at the rows of Xnew for a fitted KernelPoissonRegressor
    (est), given its training data (X, y, exposure)."""
    y = np.asarray(y, dtype=float)
    loss = est._loss(est.kappa_)
    eta = est.predict_eta(X, exposure)
    _, _, w = loss.derivatives(y, eta)
    Z = est.encoder_.transform(X)
    Zq = est.encoder_.transform(Xnew)
    lam = float(est.lam_)
    ell, fam = est.lengthscale_, est.kernel
    if est.best_["mode"] == "full":
        K = gram(Z, lengthscale=ell, family=fam)
        K[np.diag_indices_from(K)] += est.nugget
        Kq = gram(Zq, Z, lengthscale=ell, family=fam)
        sw = np.sqrt(w)
        A = sw[:, None] * K * sw[None, :]
        A[np.diag_indices_from(A)] += lam
        L = sla.cholesky(A, lower=True)
        V = sla.solve_triangular(L, sw[:, None] * Kq.T, lower=True)
        z = sla.solve_triangular(L, sw, lower=True)
        kxx = np.ones(len(Zq))          # the Gaussian and Matern kernels equal one on the diagonal
        term1 = (kxx - np.einsum("ij,ij->j", V, V)) / lam
        c = V.T @ z
        term2 = (1.0 - c) ** 2 / (lam * float(z @ z))
        return term1 + term2
    Lm = est.best_["landmarks"]
    ZL = Z[Lm]
    KnL = gram(Z, ZL, lengthscale=ell, family=fam)
    KLL = gram(ZL, ZL, lengthscale=ell, family=fam)
    m = len(Lm)
    A = np.empty((m + 1, m + 1))
    A[0, 0] = np.sum(w)
    A[0, 1:] = A[1:, 0] = KnL.T @ w
    A[1:, 1:] = KnL.T @ (w[:, None] * KnL) + lam * KLL
    A[1:, 1:][np.diag_indices(m)] += 1e-10 * lam
    C = sla.cho_factor(0.5 * (A + A.T), lower=True)
    U = np.hstack([np.ones((len(Zq), 1)), gram(Zq, ZL, lengthscale=ell, family=fam)])
    return np.einsum("ij,ji->i", U, sla.cho_solve(C, U.T))
