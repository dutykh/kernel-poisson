# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernel Poisson (or negative-binomial) regression with exposures, tuned by approximate
leave-one-out along the regularisation path.

For each length scale ell of a small grid (multiples of the median distance), one continuation
run over a decreasing geometric sequence of lam gives the whole path; at each lam the one-step
leave-one-out natural parameters eta~_i give the leave-one-out risk (mean Poisson deviance, or the
negative-binomial logarithmic score), and the pair (ell, lam) with the smallest risk is kept. The
path stops once the risk has increased for `patience` consecutive values of lam.

Up to `full_max` observations the full estimator is fitted (Newton with preconditioned conjugate
gradients, exact leverages); above it, the estimator on the span of m landmarks drawn from the
Fisher-weighted leverage scores of a pilot fit.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from .alo import alo_eta, leverages
from .kernels import Encoder, gram, median_lengthscale
from .losses import NegativeBinomial, Poisson
from .newton import KernelNewton
from .projected import ProjectedKernelNewton
from . import scores


def _risk(loss, y, eta, kappa):
    mu = np.exp(np.minimum(eta, 60.0))
    if kappa is None:
        return float(np.mean(scores.deviance(y, mu)))
    return float(np.mean(scores.log_score(y, mu, kappa)))


class KernelPoissonRegressor:
    """Kernel Poisson or negative-binomial regression with exposures, tuned by leave-one-out.

    Parameters:
    family : "poisson", or "negbin" for the negative-binomial loss, whose dispersion kappa is
        chosen together with the penalty and the length scale.
    kappas : the dispersions tried for "negbin"; by default 0.03, 0.1, 0.3 and 1 times the moment
        estimate of the dispersion of the counts (floored at 1e-3).
    kernel : "gaussian" or "matern52", on the standardised continuous covariates and the one-hot
        encoded factors (scaled by 1/sqrt(2)).
    lengthscale_factors : the length scales tried, as multiples of the median pairwise distance.
    lam_rel : the penalties tried, as penalties per observed event t, so that lam = t sum(y); by
        default 19 values from 1e-1 down to 1e-7.
    categorical : the positions of the columns of X that are factors.
    full_max : up to this number of observations the full estimator is fitted; above it the
        estimator is restricted to the span of `landmarks` landmarks.
    landmarks, pilot_landmarks : the number of landmarks of the restricted estimator, and of the
        uniform landmarks of the pilot fit whose Fisher-weighted leverage scores they are drawn from.
    patience : the path of penalties stops once the leave-one-out risk has increased for this many
        consecutive values.
    seed : the seed of every random choice (landmarks, subsample for the median distance).
    nugget : the constant added to the diagonal of the Gram matrix.
    verbose : print the leave-one-out risk along the path.

    Attributes after fit:
    lam_, lengthscale_, kappa_ : the selected penalty, length scale and dispersion (None for
        "poisson"); best_ : the selected fit, with its coefficients, intercept `b` and risk `alo`;
        path_ : the risk and the effective number of parameters `df` along every path.
    """

    def __init__(self, family="poisson", kappas=None, kernel="gaussian",
                 lengthscale_factors=(0.5, 1.0, 2.0), lam_rel=None, categorical=(),
                 full_max=6000, landmarks=1500, pilot_landmarks=500, patience=3, seed=0,
                 nugget=1e-8, verbose=False):
        self.family = family
        self.kappas = kappas
        self.kernel = kernel
        self.lengthscale_factors = tuple(lengthscale_factors)
        self.lam_rel = lam_rel if lam_rel is not None else np.logspace(-1, -7, 19)
        self.categorical = tuple(categorical)
        self.full_max = full_max
        self.landmarks = landmarks
        self.pilot_landmarks = pilot_landmarks
        self.patience = patience
        self.seed = seed
        self.nugget = nugget
        self.verbose = verbose

    # ==============================================================================
    def _loss(self, kappa):
        return Poisson() if kappa is None else NegativeBinomial(kappa)

    def _kappa_grid(self, y, offset):
        if self.family == "poisson":
            return [None]
        if self.kappas is not None:
            return list(self.kappas)
        # moment estimate around an intercept-plus-offset fit, then a grid around it
        rate = np.sum(y) / np.sum(np.exp(offset))
        mu0 = rate * np.exp(offset)
        k0 = max(float(np.sum((y - mu0) ** 2 - mu0) / np.sum(mu0 ** 2)), 1e-3)
        return sorted({float(k0 * c) for c in (0.03, 0.1, 0.3, 1.0)})

    def fit(self, X, y, exposure=None):
        """Fit on covariates X (a 2-d array), counts y and positive exposures (unit exposure by
        default); returns self."""
        y = np.asarray(y, dtype=float)
        N = len(y)
        e = np.ones(N) if exposure is None else np.asarray(exposure, dtype=float)
        o = np.log(e)
        self.encoder_ = Encoder(self.categorical).fit(X)
        Z = self.encoder_.transform(X)
        self.Z_ = Z
        ell0 = median_lengthscale(Z, seed=self.seed)
        lams = float(np.sum(y)) * np.asarray(self.lam_rel, dtype=float)
        self.path_ = []
        best = None
        rng = np.random.default_rng(self.seed)
        use_full = N <= self.full_max
        for kappa in self._kappa_grid(y, o):
            loss = self._loss(kappa)
            for c in self.lengthscale_factors:
                ell = c * ell0
                if use_full:
                    K = gram(Z, lengthscale=ell, family=self.kernel)
                    K[np.diag_indices_from(K)] += self.nugget
                    solver = KernelNewton(K, y, o, loss=loss, seed=self.seed)
                    alpha, worse, prev = None, 0, np.inf
                    for lam in sorted(lams, reverse=True):
                        fit = solver.solve(lam, alpha0=alpha, tol=1e-9)
                        alpha = fit.alpha
                        _, g, w = loss.derivatives(y, fit.eta)
                        h, hw = leverages(K, w, lam)
                        risk = _risk(loss, y, alo_eta(fit.eta, g, w, h, hw), kappa)
                        self.path_.append(dict(kappa=kappa, lengthscale=ell, lam=lam, alo=risk,
                                               df=float(np.sum(h))))
                        if self.verbose:
                            print(f"kappa={kappa} ell={ell:.3g} lam={lam:.3g} alo={risk:.5f} df={np.sum(h):.1f}")
                        if best is None or risk < best["alo"]:
                            best = dict(kappa=kappa, lengthscale=ell, lam=lam, alo=risk,
                                        alpha=fit.alpha.copy(), b=fit.b, mode="full")
                        worse = worse + 1 if risk > prev else 0
                        prev = risk
                        if worse >= self.patience:
                            break
                    del K
                else:
                    L = self._landmarks_projected(Z, y, o, loss, ell, lams, rng)
                    model = ProjectedKernelNewton(Z, y, o, loss=loss, landmarks=L,
                                                  lengthscale=ell, family=self.kernel)
                    beta, worse, prev = None, 0, np.inf
                    for lam in sorted(lams, reverse=True):
                        fit = model.solve(lam, beta0=beta, tol=1e-9)
                        beta = fit["beta"]
                        h, hw, g, w = model.leverages(fit)
                        risk = _risk(loss, y, alo_eta(fit["eta"], g, w, h, hw), kappa)
                        self.path_.append(dict(kappa=kappa, lengthscale=ell, lam=lam, alo=risk,
                                               df=float(np.sum(h))))
                        if self.verbose:
                            print(f"kappa={kappa} ell={ell:.3g} lam={lam:.3g} alo={risk:.5f} df={np.sum(h):.1f}")
                        if best is None or risk < best["alo"]:
                            best = dict(kappa=kappa, lengthscale=ell, lam=lam, alo=risk,
                                        beta=fit["beta"].copy(), b=fit["b"], landmarks=L,
                                        mode="projected")
                        worse = worse + 1 if risk > prev else 0
                        prev = risk
                        if worse >= self.patience:
                            break
                    del model
        self.best_ = best
        self.kappa_ = best["kappa"]
        self.lengthscale_ = best["lengthscale"]
        self.lam_ = best["lam"]
        return self

    def _landmarks_projected(self, Z, y, o, loss, ell, lams, rng):
        """Fisher-leverage landmarks from a pilot fit on uniform landmarks."""
        N = len(y)
        pilot = np.sort(rng.choice(N, size=min(N, self.pilot_landmarks), replace=False))
        model = ProjectedKernelNewton(Z, y, o, loss=loss, landmarks=pilot, lengthscale=ell,
                                      family=self.kernel)
        lam_mid = float(np.sort(lams)[len(lams) // 2])
        fit = model.solve(lam_mid, tol=1e-6)
        _, _, w = loss.derivatives(y, fit["eta"])
        # upper leverage estimates from the pilot columns: (1/lam)(M_ii - M_iS (M_SS + lam)^-1 M_Si)
        sw = np.sqrt(w)
        C = model.KnL * sw[:, None] * sw[pilot][None, :]
        G = C[pilot] + lam_mid * np.eye(len(pilot))
        Lc = sla.cholesky(0.5 * (G + G.T), lower=True)
        V = sla.solve_triangular(Lc, C.T, lower=True)
        est = np.clip((w * 1.0 - np.einsum("ij,ij->j", V, V)) / lam_mid, 0.0, None)
        p = est + 1e-12 * est.max()
        m = int(min(self.landmarks, N))
        return np.sort(rng.choice(N, size=m, replace=False, p=p / p.sum()))

    # ==============================================================================
    def predict_eta(self, X, exposure=None):
        """Log-mean log(exposure) + b + f(x) at the rows of X (the log-rate when exposure is None)."""
        Zq = self.encoder_.transform(X)
        o = np.zeros(len(Zq)) if exposure is None else np.log(np.asarray(exposure, dtype=float))
        b = self.best_
        if b["mode"] == "full":
            Kq = gram(Zq, self.Z_, lengthscale=self.lengthscale_, family=self.kernel)
            return o + b["b"] + Kq @ b["alpha"]
        ZL = self.Z_[b["landmarks"]]
        out = np.empty(len(Zq))
        for a in range(0, len(Zq), 20000):
            out[a:a + 20000] = gram(Zq[a:a + 20000], ZL, lengthscale=self.lengthscale_,
                                    family=self.kernel) @ b["beta"]
        return o + b["b"] + out

    def predict(self, X, exposure=None):
        """Fitted means at the rows of X (the rate per unit of exposure when exposure is None)."""
        return np.exp(np.minimum(self.predict_eta(X, exposure), 60.0))
