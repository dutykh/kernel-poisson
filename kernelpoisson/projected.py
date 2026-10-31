# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernel count regression restricted to the span of landmark kernel sections (large N).

With landmarks x_s (s in L, |L| = m), the estimator minimises over b and f in
H_L = span{K(x_s, .)},  f = sum_s beta_s K(x_s, .),

    Phi_L(b, beta) = sum_i l(y_i, o_i + b + (K_nL beta)_i) + (lam/2) beta' K_LL beta.

The intercept is profiled exactly as for the full estimator, and the profiled Newton system

    (K_nL' M K_nL + lam K_LL) dbeta = -(K_nL' g + lam K_LL beta),   M = W - w w'/(1'w),

is m x m and is solved by a Cholesky factorisation. The cross-kernel K_nL is evaluated in row
blocks, so memory grows like N m. Landmarks come from the Fisher-weighted leverage scores of a
pilot fit with uniform landmarks.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.linalg as sla

from .kernels import gram
from .losses import Poisson
from .newton import _cholesky_jitter


class ProjectedKernelNewton:
    """Profiled Newton iteration for the estimator on the landmark span."""

    def __init__(self, Z, y, offset=None, loss=None, landmarks=None, lengthscale=1.0,
                 family="gaussian", block=20000, dtype=np.float64):
        self.Z = np.asarray(Z, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.N = len(self.y)
        self.o = np.zeros(self.N) if offset is None else np.asarray(offset, dtype=float)
        self.loss = loss or Poisson()
        self.lengthscale, self.family = float(lengthscale), family
        self.L = np.asarray(landmarks)
        self.ZL = self.Z[self.L]
        self.KLL = gram(self.ZL, lengthscale=self.lengthscale, family=self.family)
        self.KnL = np.empty((self.N, len(self.L)), dtype=dtype)
        for a in range(0, self.N, block):
            self.KnL[a:a + block] = gram(self.Z[a:a + block], self.ZL,
                                         lengthscale=self.lengthscale, family=self.family)

    def _centred_gram(self, w, block=20000):
        """K_nL' M K_nL = sum_i w_i (k_i - kbar)(k_i - kbar)', kbar the w-weighted mean row, accumulated
        over row blocks as a sum of Gram matrices: positive semidefinite in floating point (no
        cancellation against w w'/sum(w)), with memory of order block x m."""
        kbar = (w @ self.KnL) / float(np.sum(w))
        m = self.KnL.shape[1]
        G = np.zeros((m, m))
        for a in range(0, self.N, block):
            C = np.sqrt(w[a:a + block])[:, None] * (self.KnL[a:a + block] - kbar[None, :])
            G += C.T @ C
        return G

    def _profile(self, f, b_start=None):
        base = self.o + f
        if self.loss.closed_form_intercept:
            return self.loss.profile_intercept(self.y, base)
        return self.loss.profile_intercept(self.y, base, b0=b_start)

    def objective(self, beta, f, b, lam):
        return self.loss.value(self.y, self.o + b + f) + 0.5 * lam * float(beta @ (self.KLL @ beta))

    def solve(self, lam, beta0=None, tol=1e-10, max_newton=60, b_start=None):
        lam = float(lam)
        m = len(self.L)
        beta = np.zeros(m) if beta0 is None else np.array(beta0, dtype=float)
        f = self.KnL @ beta
        b = self._profile(f, b_start)
        phi = self.objective(beta, f, b, lam)
        scale = max(1.0, float(np.sum(self.y)))
        records = []
        for it in range(max_newton):
            t0 = time.perf_counter()
            eta = self.o + b + f
            mu, g, w = self.loss.derivatives(self.y, eta)
            grad = self.KnL.T @ g + lam * (self.KLL @ beta)
            H = self._centred_gram(w) + lam * self.KLL
            d = -sla.cho_solve(_cholesky_jitter(0.5 * (H + H.T), 1e-12), grad, check_finite=False)
            Kd = self.KnL @ d
            slope = float(grad @ d)
            dec = -slope
            if not np.isfinite(dec) or dec <= 0.0:
                break
            step = 1.0
            while True:
                beta_new, f_new = beta + step * d, f + step * Kd
                b_new = self._profile(f_new, b)
                phi_new = self.objective(beta_new, f_new, b_new, lam)
                if phi_new <= phi + 1e-4 * step * slope or step < 1e-10:
                    break
                step *= 0.5
            beta, f, b, phi = beta_new, f_new, b_new, phi_new
            records.append(dict(lam=lam, objective=phi, decrement=dec, step=step,
                                seconds=time.perf_counter() - t0))
            if dec <= tol * scale and step == 1.0:
                break
        eta = self.o + b + f
        mu = self.loss.derivatives(self.y, eta)[0]
        return dict(lam=lam, beta=beta, b=b, f=f, eta=eta, mu=mu, objective=phi, records=records)

    def leverages(self, fit):
        """Diagonal of the local smoother of the projected model (intercept included):
        h_i = w_i d_i' (D' W D + R)^{-1} d_i,  d_i = (1, K_{i,L}),  R = diag(0, lam K_LL)."""
        lam = fit["lam"]
        _, g, w = self.loss.derivatives(self.y, fit["eta"])
        m = len(self.L)
        block = 20000
        A = np.empty((m + 1, m + 1))
        A[0, 0] = float(np.sum(w))
        A[0, 1:] = A[1:, 0] = self.KnL.T @ w
        G = np.zeros((m, m))
        for a in range(0, self.N, block):
            Kb = self.KnL[a:a + block]
            G += Kb.T @ (Kb * w[a:a + block, None])
        A[1:, 1:] = G + lam * self.KLL
        A[np.diag_indices_from(A)] += 1e-12 * float(np.trace(A)) / (m + 1)
        Lc = sla.cholesky(0.5 * (A + A.T), lower=True, check_finite=False)
        hw = np.empty(self.N)                       # d_i' (D'WD+R)^{-1} d_i, d_i = (1, K_{i,L})
        for a in range(0, self.N, block):
            Kb = self.KnL[a:a + block]
            Db = np.hstack([np.ones((len(Kb), 1)), Kb])
            V = sla.solve_triangular(Lc, Db.T, lower=True, check_finite=False)
            hw[a:a + block] = np.einsum("ij,ij->j", V, V)
        return w * hw, hw, g, w

    def predict_eta(self, fit, Znew, offset=None, block=20000):
        Znew = np.asarray(Znew, dtype=float)
        o = np.zeros(len(Znew)) if offset is None else np.asarray(offset, dtype=float)
        out = np.empty(len(Znew))
        for a in range(0, len(Znew), block):
            Kb = gram(Znew[a:a + block], self.ZL, lengthscale=self.lengthscale, family=self.family)
            out[a:a + block] = Kb @ fit["beta"]
        return o + fit["b"] + out
