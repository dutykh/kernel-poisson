# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Poisson and negative-binomial generalised additive models (R package mgcv).

    log mu(x) = log e + b + sum_j s_j(x_j) + (linear terms) + (factor effects),

with a thin-plate regression spline s(x, k = 10) for every continuous covariate having at least
ten distinct training values, a linear term for the others, factors as parametric terms, and
smoothing parameters by restricted maximum likelihood (gam, method "REML"). Above
LARGE_ABOVE training rows the same model is fitted by bam with fast REML on discretised
covariates. The negative-binomial family nb() estimates theta = 1/kappa together with the
smoothing parameters.
"""
from __future__ import annotations

import numpy as np

from . import rbridge

LARGE_ABOVE = 20000


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1, family="poisson"):
    tr, te, spec = rbridge.prepare(Xtr, ytr, etr, Xte, ete, categorical)
    large = int(len(tr) > LARGE_ABOVE)
    eta, info = rbridge.run("gam_fit.R", tr, te, spec, [family, n_threads, large], n_threads)
    mu = np.exp(np.minimum(eta, 60.0))
    kappa = None
    if family == "negbin":
        kappa = 1.0 / float(info["theta"])
    info = dict(family=family, edf=info.get("edf"), converged=bool(info.get("converged", 0)),
                fitter=info.get("fitter"), smooths=int(info.get("smooths", 0)),
                linear=int(info.get("linear", 0)), factors=int(info.get("factors", 0)),
                theta=info.get("theta") if family == "negbin" else None)
    return dict(mu=mu, kappa=kappa, info=info)
