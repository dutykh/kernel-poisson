# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Random forest with the Poisson splitting criterion (scikit-learn RandomForestRegressor).

The forest averages 500 trees grown on the rate y/e with sample weights e (the Poisson criterion
then measures the Poisson deviance of the counts in each node, and never creates a node whose
counts all vanish, so that every predicted rate is positive); the predicted mean is e times the
predicted rate. Factors enter as indicator columns. The share of covariates tried at each split
(max_features) and the minimal leaf size are chosen on GRID by 3-fold cross-validation of the
mean count deviance inside the training fold, or, above OOB_ABOVE training rows, by the
out-of-bag deviance; above MAX_SAMPLES rows each tree is grown on a bootstrap sample of
MAX_SAMPLES rows. The dispersion kappa_nb is the moment estimate from the out-of-bag means of the
final forest: the in-sample means of a forest interpolate the training counts and would
understate the dispersion.
"""
from __future__ import annotations

import numpy as np

from .common import Preprocessor, exposure, kappa_moment, kfold, mean_deviance, threadpool

GRID = [dict(max_features=mf, min_samples_leaf=s) for mf in (0.33, 0.66, 1.0) for s in (5, 20, 50, 100)]
N_TREES = 500
INNER_FOLDS = 3
OOB_ABOVE = 10000
MAX_SAMPLES = 50000


def _model(params, n, seed, n_threads, oob):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_estimators=N_TREES, criterion="poisson", bootstrap=True,
                                 oob_score=oob, max_samples=MAX_SAMPLES if n > MAX_SAMPLES else None,
                                 n_jobs=n_threads, random_state=seed, **params)


def _oob_deviance(model, y, e):
    r = model.oob_prediction_
    ok = np.isfinite(r)
    return mean_deviance(y[ok], e[ok] * np.maximum(r[ok], 1e-300)), r


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    ytr = np.asarray(ytr, dtype=float)
    n = len(ytr)
    etr_ = exposure(etr, n)
    pre = Preprocessor().fit(Xtr, categorical)
    Atr, _ = pre.matrix(Xtr, coding="full")
    Ate, _ = pre.matrix(Xte, coding="full")
    rate = ytr / etr_
    with threadpool(n_threads):
        if n <= OOB_ABOVE:
            cv = []
            for params in GRID:
                devs = []
                for tr, va in kfold(n, INNER_FOLDS, seed):
                    m = _model(params, len(tr), seed, n_threads, False).fit(
                        Atr[tr], rate[tr], sample_weight=etr_[tr])
                    devs.append(mean_deviance(ytr[va], etr_[va] * m.predict(Atr[va])))
                cv.append(float(np.mean(devs)))
            best = GRID[int(np.argmin(cv))]
            model = _model(best, n, seed, n_threads, True).fit(Atr, rate, sample_weight=etr_)
            _, oob = _oob_deviance(model, ytr, etr_)
            route = "inner 3-fold cross-validation"
        else:
            cv, model, best, oob = [], None, None, None
            for params in GRID:
                m = _model(params, n, seed, n_threads, True).fit(Atr, rate, sample_weight=etr_)
                dev, r = _oob_deviance(m, ytr, etr_)
                cv.append(dev)
                if dev == min(cv):
                    model, best, oob = m, params, r
                else:
                    del m
            route = "out-of-bag deviance"
        mu = exposure(ete, len(Ate)) * model.predict(Ate)
    ok = np.isfinite(oob)
    kappa_nb = kappa_moment(ytr[ok], etr_[ok] * oob[ok])
    info = dict(best, route=route, cv_deviance=min(cv), n_trees=N_TREES,
                max_samples=MAX_SAMPLES if n > MAX_SAMPLES else None)
    return dict(mu=mu, kappa=None, kappa_nb=kappa_nb, info=info)
