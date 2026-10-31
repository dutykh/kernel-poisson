# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Poisson and negative-binomial generalised linear models with the offset log e.

    log E[Y | x] = log e + b + x' beta,   Y ~ Poisson,  or  Y ~ NB2 with Var Y = mu + kappa mu^2,

on the standardised continuous covariates and the treatment-coded factors. The Poisson model is
fitted by iteratively reweighted least squares (statsmodels GLM); the negative-binomial model by
maximum likelihood in (b, beta, kappa) jointly (statsmodels NegativeBinomial, loglike_method
"nb2"), started from the Poisson fit and the moment estimate of kappa. Indicator columns that
are linear combinations of others on the training fold (for example workingday against weekday
and holiday) are removed before fitting. The coefficients are also reported on the original
scale of the covariates: beta_j / s_j, and b - sum_j beta_j m_j / s_j for the intercept.
"""
from __future__ import annotations

import warnings

import numpy as np

from .common import Preprocessor, exposure, independent_columns, kappa_moment, threadpool


def _design(pre, X):
    A, names = pre.matrix(X, coding="treatment")
    return np.hstack([np.ones((len(A), 1)), A]), ["(intercept)"] + names


def _original_scale(pre, names, beta):
    coef = dict(zip(names, map(float, beta)))
    b0 = coef.pop("(intercept)", 0.0)
    out = {}
    for c in pre.continuous_:
        if c in coef:
            out[c] = coef[c] / pre.scale_[c]
            b0 -= coef[c] * pre.mean_[c] / pre.scale_[c]
    for k, v in coef.items():
        if k not in out:
            out[k] = v
    return b0, out


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1, family="poisson"):
    import statsmodels.api as sm
    from statsmodels.discrete.discrete_model import NegativeBinomial

    ytr = np.asarray(ytr, dtype=float)
    otr = np.log(exposure(etr, len(ytr)))
    pre = Preprocessor().fit(Xtr, categorical)
    Atr, names = _design(pre, Xtr)
    Ate, _ = _design(pre, Xte)
    keep = independent_columns(Atr)
    dropped = [names[j] for j in range(len(names)) if j not in set(keep.tolist())]
    Atr, Ate, names = Atr[:, keep], Ate[:, keep], [names[j] for j in keep]
    ote = np.log(exposure(ete, len(Ate)))
    with threadpool(n_threads), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pois = sm.GLM(ytr, Atr, family=sm.families.Poisson(), offset=otr).fit(tol=1e-10, maxiter=200)
        beta = np.asarray(pois.params)
        kappa = None
        converged = bool(pois.converged)
        if family == "negbin":
            mu0 = np.exp(Atr @ beta + otr)
            start = np.append(beta, kappa_moment(ytr, mu0))
            model = NegativeBinomial(ytr, Atr, loglike_method="nb2", offset=otr)
            res = None
            for method in ("newton", "bfgs"):
                try:
                    res = model.fit(start_params=start, method=method, maxiter=500, disp=0)
                    if np.all(np.isfinite(res.params)):
                        break
                except Exception:
                    res = None
            if res is None:
                raise RuntimeError("negative-binomial likelihood maximisation failed")
            params = np.asarray(res.params)
            beta, kappa = params[:-1], float(params[-1])
            converged = bool(res.mle_retvals.get("converged", True))
    mu = np.exp(np.minimum(Ate @ beta + ote, 60.0))
    b0, coef = _original_scale(pre, names, beta)
    info = dict(family=family, n_coef=len(beta), dropped_collinear=dropped, converged=converged,
                intercept=b0, coef=coef)
    return dict(mu=mu, kappa=kappa, info=info)
