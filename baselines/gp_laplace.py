# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Gaussian-process Poisson regression with hyperparameters chosen by the Laplace evidence.

The model is that of the kernel Poisson estimator: y_i ~ Poisson(e_i exp(b + f(x_i))) with the
Gaussian kernel k_ell on the encoded covariates, now with the prior f ~ GP(0, k_ell / lam) and
the intercept b a free parameter. For fixed (ell, lam) the posterior mode is the penalised fit
(b, f = K alpha) minimising

    Phi(b, alpha) = sum_i [exp(eta_i) - y_i eta_i] + (lam/2) alpha' K alpha,  eta = log e + b + K alpha,

computed by the profiled Newton iteration of kernelpoisson (KernelNewton). Expanding the log
posterior to second order in f about the mode gives the Laplace approximation of the log
marginal likelihood, up to terms free of (ell, lam),

    log p(y | ell, lam) ~ -Phi(b, alpha) - (1/2) log det(I + W^{1/2} K W^{1/2} / lam),
    W = diag(mu),  mu = exp(eta),

whose log-determinant is twice the sum of the logarithms of the diagonal of the Cholesky factor
of I + W^{1/2} K W^{1/2} / lam. The pair (ell, lam) maximising it over the grid (ell a multiple
of the median pairwise distance, lam decreasing along LAMS with warm starts, a path stopped once
the evidence has fallen for PATIENCE consecutive values) gives the predictions. When the landmark
preconditioner of the Newton iteration fails (its Cholesky factorisation loses definiteness once
lam is small against the Fisher weights sum_i mu_i), the rest of that path is solved by
conjugate gradients without it. Each grid point costs one O(N^3) factorisation, so the method
is not run above MAX_TRAIN training rows.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .common import as_frame, exposure, factor_names, threadpool

MAX_TRAIN = 5000
LENGTHSCALE_FACTORS = (0.25, 0.5, 1.0, 2.0, 4.0)
LAMS = 10.0 ** np.linspace(3.0, -4.0, 15)
PATIENCE = 3
NUGGET = 1e-8


def log_evidence(K, mu, lam, objective):
    """Laplace approximation -Phi - (1/2) log det(I + W^{1/2} K W^{1/2} / lam)."""
    sw = np.sqrt(mu)
    B = (sw[:, None] * K) * (sw[None, :] / lam)
    B[np.diag_indices_from(B)] += 1.0
    L = sla.cholesky(B, lower=True, overwrite_a=True, check_finite=False)
    return -objective - float(np.sum(np.log(np.diag(L))))


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    from kernelpoisson import KernelNewton
    from kernelpoisson.kernels import Encoder, gram, median_lengthscale

    ytr = np.asarray(ytr, dtype=float)
    if len(ytr) > MAX_TRAIN:
        return None
    Xtr, Xte = as_frame(Xtr), as_frame(Xte)
    idx = [Xtr.columns.get_loc(c) for c in factor_names(Xtr, categorical)]
    enc = Encoder(idx).fit(Xtr.to_numpy(dtype=object))
    Ztr = enc.transform(Xtr.to_numpy(dtype=object))
    Zte = enc.transform(Xte.to_numpy(dtype=object))
    otr = np.log(exposure(etr, len(ytr)))
    ote = np.log(exposure(ete, len(Zte)))
    ell0 = median_lengthscale(Ztr, seed=seed)
    best, path, fallbacks = None, [], 0
    with threadpool(n_threads):
        for c in LENGTHSCALE_FACTORS:
            ell = c * ell0
            K = gram(Ztr, lengthscale=ell)
            K[np.diag_indices_from(K)] += NUGGET
            solver = KernelNewton(K, ytr, otr, seed=seed)
            alpha, prev, worse = None, -np.inf, 0
            for lam in LAMS:
                try:
                    fit = solver.solve(lam, alpha0=alpha, tol=1e-9)
                except np.linalg.LinAlgError:
                    # the landmark preconditioner can lose definiteness when lam is small
                    # against the Fisher weights; conjugate gradients without it are used then
                    solver = KernelNewton(K, ytr, otr, seed=seed, preconditioner="none")
                    fit = solver.solve(lam, alpha0=alpha, tol=1e-9)
                    fallbacks += 1
                alpha = fit.alpha
                ev = log_evidence(K, fit.mu, lam, fit.objective)
                path.append(dict(lengthscale=ell, lam=float(lam), log_evidence=ev))
                if best is None or ev > best["log_evidence"]:
                    best = dict(lengthscale=ell, factor=c, lam=float(lam), log_evidence=ev,
                                alpha=fit.alpha.copy(), b=float(fit.b))
                worse = worse + 1 if ev < prev else 0
                prev = ev
                if worse >= PATIENCE:
                    break
            del K, solver
        eta = ote + best["b"] + gram(Zte, Ztr, lengthscale=best["lengthscale"]) @ best["alpha"]
    mu = np.exp(np.minimum(eta, 60.0))
    info = dict(lengthscale=best["lengthscale"], lengthscale_factor=best["factor"],
                lam=best["lam"], log_evidence=best["log_evidence"], grid_points=len(path),
                unpreconditioned_paths=fallbacks,
                at_grid_edge=bool(best["lam"] in (LAMS[0], LAMS[-1])
                                  or best["factor"] in (LENGTHSCALE_FACTORS[0],
                                                        LENGTHSCALE_FACTORS[-1])))
    return dict(mu=mu, kappa=None, info=info)
