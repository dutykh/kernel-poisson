# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Landmarks drawn from Fisher-weighted ridge leverage scores.

For weights w_i = l''(y_i, eta_i) (the Fisher information of observation i about its log-rate;
w_i = mu_i for the Poisson model) let M = W^{1/2} K W^{1/2}. The Fisher-weighted ridge leverage
score of observation i at level lam is

    ell_i(lam) = [M (M + lam I)^{-1}]_ii = (1/lam) (M_ii - M_{i,:} (M + lam I)^{-1} M_{:,i}),

and d(lam) = sum_i ell_i(lam) is the effective dimension of the fitted smoother. Restricting the
quadratic form to a pilot subset S0 can only decrease it, so

    ell~_i = (1/lam) (M_ii - M_{i,S0} (M_{S0,S0} + lam I)^{-1} M_{S0,i}) ⩾ ell_i(lam)

is an upper estimate computable in O(N |S0|^2); sampling in proportion to it is the rule
behind the landmark guarantee of the preconditioner.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla


def leverage_upper(K, w, lam, pilot):
    """Upper estimates of the Fisher-weighted ridge leverage scores from a pilot subset."""
    sw = np.sqrt(np.maximum(w, 0.0))
    pilot = np.asarray(pilot)
    C = K[:, pilot] * sw[:, None] * sw[pilot][None, :]          # M_{:, S0}
    G = C[pilot, :] + lam * np.eye(len(pilot))                   # M_{S0,S0} + lam I
    L = sla.cholesky(G, lower=True)
    V = sla.solve_triangular(L, C.T, lower=True)                 # L^{-1} M_{S0,:}
    diag_M = w * np.diag(K)
    est = (diag_M - np.einsum("ij,ij->j", V, V)) / lam
    return np.clip(est, 0.0, None)


def fisher_landmarks(K, w, lam, m, rng, pilot_size=None, pilot=None):
    """Sample m distinct landmarks with probabilities proportional to the upper leverage
    estimates; returns (landmarks, estimates, estimated effective dimension)."""
    N = K.shape[0]
    m = int(min(max(m, 1), N))
    if pilot is None:
        size = int(min(N, pilot_size or max(2 * m, 200)))
        pilot = rng.choice(N, size=size, replace=False)
    est = leverage_upper(K, w, lam, pilot)
    p = est + 1e-12 * est.max()
    p = p / p.sum()
    chosen = rng.choice(N, size=m, replace=False, p=p)
    return np.sort(chosen), est, float(np.sum(np.minimum(est, 1.0)))


def uniform_landmarks(N, m, rng):
    return np.sort(rng.choice(N, size=int(min(m, N)), replace=False))
