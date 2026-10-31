# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Newton's method for kernel count regression with a profiled intercept.

Problem. For counts y_i, offsets o_i = log e_i, a bounded positive definite kernel K with Gram
matrix K, and lam > 0, minimise over the intercept b and f = sum_j alpha_j K(x_j, .)

    Phi(b, alpha) = sum_i l(y_i, o_i + b + (K alpha)_i) + (lam/2) alpha' K alpha.

The intercept is profiled: b(alpha) minimises Phi(., alpha) (closed form for the Poisson loss),
so every iterate satisfies the intercept score equation sum_i l'(y_i, eta_i) = 0, which for the
Poisson loss is the balance sum_i mu_i = sum_i y_i.

Newton step. With g = l'(y, eta), w = l''(y, eta) > 0 and M = W - w w'/(1'w) (the centred
Fisher weights), the profiled Newton direction solves

    K (M K + lam I) dalpha = -K r,        r = g + lam alpha,

which is the linear system of the truncated-Newton method of Maalouf, Trafalis and Adrianto
(2011) with the intercept eliminated. For the element df = sum_j dalpha_j K(x_j, .) it reads
(S* M S + lam I) df = -grad, with S the evaluation operator at the data. Conjugate gradients
is run in the inner product of the reproducing-kernel space, <J c, J c'> = c' K c, and is
preconditioned by

    P = Pi S* M S Pi + lam I,   Pi = orthogonal projection onto span{K(x_s, .), s in L},

for landmarks L. Each iteration costs one product with K and O(N m) operations; every truncated
iterate decreases the quadratic model, hence is a descent direction.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla

from .landmarks import leverage_upper, uniform_landmarks
from .losses import Poisson


# ======================================================================================
def _cholesky_jitter(A, jitter, tries=8):
    """Cholesky factor of a symmetric positive semidefinite matrix, adding jitter times the mean
    diagonal, multiplied by ten after each failure."""
    m = A.shape[0]
    scale = float(np.mean(np.diag(A))) if m else 1.0
    eps = jitter * max(scale, np.finfo(float).tiny)
    for _ in range(tries):
        try:
            return sla.cho_factor(A + eps * np.eye(m), lower=True, check_finite=False)
        except np.linalg.LinAlgError:
            eps *= 10.0
    return sla.cho_factor(A + eps * np.eye(m), lower=True, check_finite=False)


class ProjectedPreconditioner:
    """Inverse of P = Pi S* M S Pi + lam I applied to coefficient vectors.

    For g = J z with values g_L = (K z)_L at the landmarks,
        Pi g = sum_s gamma_s K(x_s, .),               K_LL gamma = g_L,
        (P restricted to the landmark span)^{-1} Pi g = sum_s beta_s K(x_s, .),
                                                       (lam K_LL + K_nL' M K_nL) beta = g_L,
        P^{-1} g = (g - Pi g)/lam + sum_s beta_s K(x_s, .).
    """

    def __init__(self, K, w, lam, landmarks, jitter=1e-10):
        L = np.asarray(landmarks)
        self.L, self.lam = L, float(lam)
        KnL = K[:, L]
        self.KnL = KnL
        KLL = np.asarray(KnL[L, :])
        # K_nL' M K_nL = sum_i w_i (k_i - kbar)(k_i - kbar)', formed as a Gram matrix so that it
        # stays positive semidefinite in floating point (no cancellation against w w'/sum(w))
        kbar = (w @ KnL) / float(np.sum(w))
        C = np.sqrt(w)[:, None] * (np.asarray(KnL) - kbar[None, :])
        Q = self.lam * KLL + C.T @ C
        self.cK = _cholesky_jitter(0.5 * (KLL + KLL.T), jitter)
        self.cQ = _cholesky_jitter(0.5 * (Q + Q.T), jitter)

    def apply(self, z):
        gL = self.KnL.T @ z
        gam = sla.cho_solve(self.cK, gL, check_finite=False)
        bet = sla.cho_solve(self.cQ, gL, check_finite=False)
        out = z / self.lam
        out[self.L] += bet - gam / self.lam
        return out


class IdentityPreconditioner:
    """No preconditioning beyond the reproducing-kernel inner product (P = I on H_K)."""

    def apply(self, z):
        return z.copy()


# ======================================================================================
def rkhs_pcg(K, w, lam, r, precond, forcing, max_iter):
    """Solve (M K + lam I) d = -r in the inner product <c, c'> = c' K c.

    Returns d, K d, the number of iterations and the final relative preconditioned residual.
    """
    sw = float(np.sum(w))
    d = np.zeros_like(r)
    Kd = np.zeros_like(r)
    z = -r
    zeta = precond.apply(z)
    Kzeta = K @ zeta
    gamma = float(z @ Kzeta)
    gamma0 = gamma
    if gamma0 <= 0.0:
        return d, Kd, 0, 0.0
    p, Kp = zeta, Kzeta
    k = 0
    rel = 1.0
    for k in range(1, max_iter + 1):
        q = w * Kp - w * (float(w @ Kp) / sw) + lam * p
        pHp = float(Kp @ q)
        if pHp <= 0.0:
            break
        a = gamma / pHp
        d = d + a * p
        Kd = Kd + a * Kp
        z = z - a * q
        zeta = precond.apply(z)
        Kzeta = K @ zeta
        gamma_new = float(z @ Kzeta)
        rel = np.sqrt(max(gamma_new, 0.0) / gamma0)
        if rel <= forcing:
            break
        beta = gamma_new / gamma
        gamma = gamma_new
        p = zeta + beta * p
        Kp = Kzeta + beta * Kp
    return d, Kd, k, rel


def exact_newton_direction(K, w, lam, r):
    """Reference: solve (M K + lam I) d = -r by an LU factorisation (O(N^3))."""
    N = len(r)
    sw = float(np.sum(w))
    A = w[:, None] * K - np.outer(w, w @ K) / sw + lam * np.eye(N)
    d = sla.solve(A, -r, check_finite=False)
    return d, K @ d


# ======================================================================================
@dataclass
class NewtonRecord:
    lam: float
    objective: float
    decrement: float
    step: float
    cg_iterations: int
    cg_residual: float
    landmarks: int
    effective_dimension: float
    seconds: float


@dataclass
class Fit:
    """State of a fitted kernel count regression at one value of lam."""
    lam: float
    alpha: np.ndarray
    f: np.ndarray            # K alpha, the fitted log-rate contributions at the data
    b: float
    eta: np.ndarray
    mu: np.ndarray
    objective: float
    records: list = field(default_factory=list)
    landmarks: np.ndarray | None = None


class KernelNewton:
    """Profiled Newton iteration for sum_i l(y_i, o_i + b + f(x_i)) + (lam/2)|f|^2."""

    def __init__(self, K, y, offset=None, loss=None, *, preconditioner="fisher",
                 landmark_factor=3.0, max_landmarks=4000, forcing_max=0.5, max_cg=300,
                 seed=0, refresh_every=1000):
        self.K = K
        self.y = np.asarray(y, dtype=float)
        self.N = len(self.y)
        self.o = np.zeros(self.N) if offset is None else np.asarray(offset, dtype=float)
        self.loss = loss or Poisson()
        self.preconditioner = preconditioner
        self.landmark_factor = landmark_factor
        self.max_landmarks = max_landmarks
        self.forcing_max = forcing_max
        self.max_cg = max_cg
        self.rng = np.random.default_rng(seed)
        self.refresh_every = refresh_every
        self._b = None

    # ==== profiled objective ========================================================
    def _profile(self, f, b_start=None):
        base = self.o + f
        if self.loss.closed_form_intercept:
            return self.loss.profile_intercept(self.y, base)
        return self.loss.profile_intercept(self.y, base, b0=b_start)

    def objective(self, alpha, f, b):
        return self.loss.value(self.y, self.o + b + f) + 0.5 * self._lam * float(alpha @ f)

    # ==== landmarks and preconditioner ==============================================
    def _landmarks(self, w, lam):
        """Landmarks for the preconditioner: m = landmark_factor x (estimated effective
        dimension), drawn with probabilities proportional to the upper leverage estimates
        (or uniformly, for comparison)."""
        if self.preconditioner in ("none", "identity"):
            return None, float("nan")
        pilot_size = min(self.N, 600)
        while True:
            pilot = self.rng.choice(self.N, size=pilot_size, replace=False)
            est = leverage_upper(self.K, w, lam, pilot)
            dhat = float(np.sum(np.minimum(est, 1.0)))
            m = int(min(max(int(np.ceil(self.landmark_factor * dhat)), 20),
                        self.max_landmarks, self.N))
            # the pilot must be large compared with the effective dimension it resolves
            if pilot_size >= min(self.N, 2 * m):
                break
            pilot_size = min(self.N, 2 * m)
        if self.preconditioner == "uniform":
            return uniform_landmarks(self.N, m, self.rng), dhat
        p = est + 1e-12 * est.max()
        L = np.sort(self.rng.choice(self.N, size=m, replace=False, p=p / p.sum()))
        return L, dhat

    def _precond(self, w, lam, L):
        if L is None:
            return IdentityPreconditioner()
        return ProjectedPreconditioner(self.K, w, lam, L)

    # ==== one value of lam ============================================================
    def solve(self, lam, alpha0=None, tol=1e-10, max_newton=60, exact=False, landmarks=None):
        self._lam = float(lam)
        y, K = self.y, self.K
        alpha = np.zeros(self.N) if alpha0 is None else np.array(alpha0, dtype=float)
        f = K @ alpha
        b = self._profile(f, self._b)
        phi = self.objective(alpha, f, b)
        records = []
        L, dhat = landmarks, float("nan")
        scale = max(1.0, float(np.sum(y)))
        forcing_prev = None
        dec_prev = None
        for it in range(max_newton):
            t0 = time.perf_counter()
            eta = self.o + b + f
            mu, g, w = self.loss.derivatives(y, eta)
            r = g + self._lam * alpha
            if exact:
                d, Kd = exact_newton_direction(K, w, self._lam, r)
                ncg, res = 0, 0.0
            else:
                if L is None or (it > 0 and it % self.refresh_every == 0):
                    L, dhat = self._landmarks(w, self._lam)
                P = self._precond(w, self._lam, L)
                # Eisenstat-Walker choice 2 for the forcing term, safeguarded
                if forcing_prev is None or dec_prev is None:
                    forcing = self.forcing_max
                else:
                    forcing = forcing_prev
                d, Kd, ncg, res = rkhs_pcg(K, w, self._lam, r, P, forcing, self.max_cg)
            slope = float(r @ Kd)          # directional derivative of the profiled objective
            dec = -slope
            if not np.isfinite(dec) or dec <= 0.0:
                records.append(NewtonRecord(self._lam, phi, dec, 0.0, ncg, res,
                                            0 if L is None else len(L), dhat,
                                            time.perf_counter() - t0))
                break
            # backtracking on the profiled objective (Armijo, sigma = 1e-4)
            step = 1.0
            while True:
                a_new = alpha + step * d
                f_new = f + step * Kd
                b_new = self._profile(f_new, b)
                phi_new = self.objective(a_new, f_new, b_new)
                if phi_new <= phi + 1e-4 * step * slope or step < 1e-10:
                    break
                step *= 0.5
            alpha, f, b, phi_old, phi = a_new, f_new, b_new, phi, phi_new
            records.append(NewtonRecord(self._lam, phi, dec, step, ncg, res,
                                        0 if L is None else len(L), dhat,
                                        time.perf_counter() - t0))
            if dec_prev is not None:
                forcing_prev = float(np.clip(0.9 * (dec / dec_prev), 1e-8, self.forcing_max))
            dec_prev = dec
            if dec <= tol * scale and step == 1.0:
                break
        self._b = b
        eta = self.o + b + f
        mu = self.loss.derivatives(y, eta)[0]
        return Fit(lam=self._lam, alpha=alpha, f=f, b=b, eta=eta, mu=mu, objective=phi,
                   records=records, landmarks=L)

    # ==== continuation in lam =========================================================
    def path(self, lams, tol_path=1e-6, tol_final=1e-10, max_newton=60, exact=False):
        """Solve for a decreasing sequence of lam, warm-starting each value from the previous
        one; intermediate values are solved to a looser tolerance."""
        lams = sorted(map(float, lams), reverse=True)
        fits, alpha = [], None
        for k, lam in enumerate(lams):
            tol = tol_final if k == len(lams) - 1 else tol_path
            fit = self.solve(lam, alpha0=alpha, tol=tol, max_newton=max_newton, exact=exact)
            fits.append(fit)
            alpha = fit.alpha
        return fits
