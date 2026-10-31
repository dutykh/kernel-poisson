# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Kernel ridge regression of the log-transformed rate, the least-squares route to kernel count
regression that the likelihood-based estimators are compared with.

The response t = log(1 + y/e) is regressed on the encoded covariates (those of the kernel Poisson
estimator) with the Gaussian kernel exp(-|z - z'|^2 / (2 ell^2)) (scikit-learn KernelRidge),
after subtraction of its weighted mean, with sample weights proportional to e (the precision of
a rate grows with its exposure); the exposures are first expressed in units of the training
mean rate, e <- e sum(y) / sum(e), which makes the method independent of the unit of exposure. The
prediction is back-transformed as mu = e max(exp(t^) - 1, 0.01), a floor that keeps the predicted
law proper. The ridge parameter alpha and the length scale (a multiple of the median pairwise
distance) are chosen by 3-fold cross-validation of the mean count deviance inside the training
fold. The dense N x N kernel system is solved, so the method is not run above MAX_TRAIN
training rows.
"""
from __future__ import annotations

import numpy as np

from .common import as_frame, exposure, factor_names, kappa_moment, kfold, mean_deviance, threadpool

MAX_TRAIN = 8000
LENGTHSCALE_FACTORS = (0.5, 1.0, 2.0)
ALPHAS = 10.0 ** np.arange(-3.0, 3.0)
INNER_FOLDS = 3
FLOOR_SHARE = 0.01


def _fit(K, t, w, alpha):
    from sklearn.kernel_ridge import KernelRidge
    tbar = float(np.sum(w * t) / np.sum(w))
    return KernelRidge(alpha=alpha, kernel="precomputed").fit(K, t - tbar, sample_weight=w), tbar


def _means(model, tbar, Kq, e, floor):
    return e * np.maximum(np.expm1(model.predict(Kq) + tbar), floor)


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    from kernelpoisson.kernels import Encoder, gram, median_lengthscale

    ytr = np.asarray(ytr, dtype=float)
    n = len(ytr)
    if n > MAX_TRAIN:
        return None
    Xtr, Xte = as_frame(Xtr), as_frame(Xte)
    idx = [Xtr.columns.get_loc(c) for c in factor_names(Xtr, categorical)]
    enc = Encoder(idx).fit(Xtr.to_numpy(dtype=object))
    Ztr = enc.transform(Xtr.to_numpy(dtype=object))
    Zte = enc.transform(Xte.to_numpy(dtype=object))
    etr_, ete_ = exposure(etr, n), exposure(ete, len(Zte))
    # exposures in units of the training mean rate, so that the transform log(1 + y/e) and the fit
    # do not depend on the unit in which the exposure is recorded
    rbar = float(np.sum(ytr) / np.sum(etr_))
    etr_, ete_ = etr_ * rbar, ete_ * rbar
    t = np.log1p(ytr / etr_)
    w = etr_ / np.mean(etr_)
    floor = FLOOR_SHARE
    ell0 = median_lengthscale(Ztr, seed=seed)
    folds = kfold(n, INNER_FOLDS, seed)
    with threadpool(n_threads):
        scores = {}
        for c in LENGTHSCALE_FACTORS:
            K = gram(Ztr, lengthscale=c * ell0)
            for tr, va in folds:
                Ktr, Kva = K[np.ix_(tr, tr)], K[np.ix_(va, tr)]
                for a in ALPHAS:
                    model, tbar = _fit(Ktr, t[tr], w[tr], a)
                    dev = mean_deviance(ytr[va], _means(model, tbar, Kva, etr_[va], floor))
                    scores.setdefault((c, float(a)), []).append(dev)
            del K
        cv = {k: float(np.mean(v)) for k, v in scores.items()}
        c, a = min(cv, key=cv.get)
        K = gram(Ztr, lengthscale=c * ell0)
        model, tbar = _fit(K, t, w, a)
        mu = _means(model, tbar, gram(Zte, Ztr, lengthscale=c * ell0), ete_, floor)
        mu_in = _means(model, tbar, K, etr_, floor)
    info = dict(lengthscale=c * ell0, lengthscale_factor=c, alpha=a, cv_deviance=cv[(c, a)],
                floor_rate=floor)
    return dict(mu=mu, kappa=None, kappa_nb=kappa_moment(ytr, mu_in), info=info)
