# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernel Tweedie regression (R package ktweedie) with power rho = 1.01.

The package models a semicontinuous (compound Poisson-gamma) response: the log-mean is a function
f = K alpha of the reproducing-kernel space of the Gaussian kernel exp(-sigma |z - z'|^2) on the
encoded covariates z (standardised continuous columns, factor indicators scaled by 1/sqrt 2, as
for the kernel Poisson estimator), with no intercept and the penalty lambda alpha' K alpha. It
accepts neither offsets nor weights, so the response is the rate y/e divided by the training
mean rate r = sum y / sum e, and the predicted mean is e r exp(f(x)): the factor r plays the
part of the missing intercept (a Tweedie variable divided by a constant is again Tweedie, with
the mean divided by it), so that the fit does not depend on the unit of the exposure. The
bandwidth is the median heuristic sigma = 1/(2 m^2), m the median pairwise
distance of the training inputs; lambda is chosen by ktd_cv (5-fold Tweedie log-likelihood) on
LAMBDAS. The full N x N kernel matrix is formed, so the method is not run above MAX_TRAIN
training rows.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from .common import as_frame, exposure, factor_names, read_r_info, run_rscript

MAX_TRAIN = 3000
RHO = 1.01
NFOLDS = 5
LAMBDAS = 10.0 ** np.arange(-7, 1)


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    from kernelpoisson.kernels import Encoder, median_lengthscale

    ytr = np.asarray(ytr, dtype=float)
    if len(ytr) > MAX_TRAIN:
        return None
    Xtr, Xte = as_frame(Xtr), as_frame(Xte)
    cats = factor_names(Xtr, categorical)
    idx = [Xtr.columns.get_loc(c) for c in cats]
    enc = Encoder(idx).fit(Xtr.to_numpy(dtype=object))
    Ztr = enc.transform(Xtr.to_numpy(dtype=object))
    Zte = enc.transform(Xte.to_numpy(dtype=object))
    ell = median_lengthscale(Ztr, seed=seed)
    sigma = 1.0 / (2.0 * ell ** 2)
    etr_, ete_ = exposure(etr, len(ytr)), exposure(ete, len(Zte))
    rbar = float(np.sum(ytr) / np.sum(etr_))
    with tempfile.TemporaryDirectory(prefix="ktweedie-") as tmp:
        folder = Path(tmp)
        cols = [f"z{j + 1}" for j in range(Ztr.shape[1])]
        pd.DataFrame(Ztr, columns=cols).to_csv(folder / "xtr.csv", index=False)
        pd.DataFrame(Zte, columns=cols).to_csv(folder / "xte.csv", index=False)
        pd.DataFrame({"y": ytr / etr_ / rbar}).to_csv(folder / "ytr.csv", index=False)
        pd.DataFrame({"lambda": LAMBDAS}).to_csv(folder / "lambda.csv", index=False)
        pd.DataFrame({"sigma": [sigma], "rho": [RHO], "nfolds": [NFOLDS]}).to_csv(
            folder / "par.csv", index=False)
        run_rscript("ktweedie_fit.R", [folder, int(seed) + 1], n_threads, cwd=folder)
        eta = pd.read_csv(folder / "pred.csv")["eta"].to_numpy(dtype=float)
        info = read_r_info(folder / "info.csv")
    mu = ete_ * rbar * np.exp(np.minimum(eta, 60.0))
    info = dict(response="semicontinuous (Tweedie compound Poisson-gamma) method, fitted to the "
                         "rate y/e over the training mean rate, without offset or weights",
                mean_rate=rbar, rho=RHO, sigma=sigma,
                lengthscale=ell, lam=info["lambda"], cv_loglik=info["cv_loglik"],
                convergence=info["convergence"], route=info["route"])
    return dict(mu=mu, kappa=None, info=info)
