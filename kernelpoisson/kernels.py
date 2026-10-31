# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernels on mixed inputs and their Gram matrices.

Continuous covariates are standardised; a categorical covariate with levels a_1, ..., a_L is
encoded by the L unit vectors scaled by 1/sqrt(2), so that two observations that differ in
that factor are at squared distance exactly one. On the encoded vectors z,

    Gaussian:  k(z, z') = exp(-|z - z'|^2 / (2 ell^2)),
    Matern-5/2: k(z, z') = (1 + sqrt5 r/ell + 5 r^2/(3 ell^2)) exp(-sqrt5 r/ell),  r = |z - z'|,

so a mismatch in one factor multiplies the Gaussian kernel by exp(-1/(2 ell^2)): the Gaussian
kernel on the encoded inputs is the product of a Gaussian kernel on the continuous part and an
exponential kernel of the Hamming distance on the factors, with a single length scale ell.
Both kernels are bounded by one on the diagonal.
"""
from __future__ import annotations

import numpy as np


class Encoder:
    """Standardise continuous columns and one-hot encode categorical columns (scale 1/sqrt 2)."""

    def __init__(self, categorical=()):
        self.categorical = tuple(categorical)

    def fit(self, X):
        X = np.asarray(X, dtype=object if self.categorical else float)
        p = X.shape[1]
        self.continuous_ = [j for j in range(p) if j not in self.categorical]
        Xc = np.asarray(X[:, self.continuous_], dtype=float) if self.continuous_ else np.zeros((len(X), 0))
        self.mean_ = Xc.mean(axis=0) if Xc.size else np.zeros(0)
        sd = Xc.std(axis=0) if Xc.size else np.zeros(0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        self.levels_ = {j: sorted(set(X[:, j].tolist()), key=str) for j in self.categorical}
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=object if self.categorical else float)
        blocks = []
        if self.continuous_:
            Xc = np.asarray(X[:, self.continuous_], dtype=float)
            blocks.append((Xc - self.mean_) / self.scale_)
        for j in self.categorical:
            levels = self.levels_[j]
            col = X[:, j].tolist()
            onehot = np.zeros((len(col), len(levels)))
            index = {v: i for i, v in enumerate(levels)}
            for r, v in enumerate(col):
                if v in index:  # an unseen level is equidistant from every seen level
                    onehot[r, index[v]] = 1.0
            blocks.append(onehot / np.sqrt(2.0))
        return np.hstack(blocks) if blocks else np.zeros((len(X), 0))

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def squared_distances(A, B=None):
    """Matrix of |a_i - b_j|^2, clipped at zero against rounding."""
    A = np.asarray(A, dtype=float)
    B = A if B is None else np.asarray(B, dtype=float)
    aa = np.einsum("ij,ij->i", A, A)[:, None]
    bb = np.einsum("ij,ij->i", B, B)[None, :]
    d2 = aa + bb - 2.0 * (A @ B.T)
    np.maximum(d2, 0.0, out=d2)
    return d2


def gram(A, B=None, lengthscale=1.0, family="gaussian"):
    """Gram matrix k(a_i, b_j) of the Gaussian or Matern-5/2 kernel."""
    d2 = squared_distances(A, B)
    if family == "gaussian":
        d2 *= -0.5 / lengthscale ** 2
        return np.exp(d2, out=d2)
    if family == "matern52":
        r = np.sqrt(d2) * (np.sqrt(5.0) / lengthscale)
        return (1.0 + r + r * r / 3.0) * np.exp(-r)
    raise ValueError(f"unknown kernel family {family!r}")


def median_lengthscale(Z, max_points=2000, seed=0):
    """Median of the pairwise distances, the usual scale-free reference length."""
    Z = np.asarray(Z, dtype=float)
    if len(Z) > max_points:
        rng = np.random.default_rng(seed)
        Z = Z[rng.choice(len(Z), max_points, replace=False)]
    d2 = squared_distances(Z)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(np.sqrt(d2[iu])))
    return med if med > 0 else 1.0
