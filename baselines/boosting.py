# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Gradient-boosted regression trees with the Poisson loss (scikit-learn HistGradientBoosting).

The trees model the log-rate: the response is the rate y/e with sample weights e, so that the
weighted Poisson deviance of the rates, sum_i e_i d(y_i/e_i, r_i), equals the Poisson deviance
sum_i d(y_i, e_i r_i) of the counts; the predicted mean is e times the predicted rate. Factors
are handled natively (categorical splits). Each fit stops early when the Poisson loss on an
internal 15 per cent validation split has not improved for 20 iterations (at most 2000). The
learning rate, the number of leaves and the minimal leaf size are chosen on GRID by 3-fold
cross-validation of the mean count deviance inside the training fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import Preprocessor, exposure, kappa_moment, kfold, mean_deviance, threadpool

GRID = [dict(learning_rate=lr, max_leaf_nodes=m, min_samples_leaf=s)
        for lr in (0.05, 0.1) for m in (4, 8, 16, 31) for s in (20, 100)]
INNER_FOLDS = 3


def _frame(pre, X):
    Xc = pre.clean(X)
    out = pd.DataFrame(index=range(len(Xc)))
    for c in pre.continuous_:
        out[c] = Xc[c].to_numpy(dtype=float)
    for c in pre.factors_:
        out[c] = pd.Categorical(Xc[c].to_numpy(dtype=object), categories=pre.levels_[c])
    return out


def _model(params, seed):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(loss="poisson", max_iter=2000, early_stopping=True,
                                         validation_fraction=0.15, n_iter_no_change=20,
                                         categorical_features="from_dtype", random_state=seed,
                                         **params)


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    ytr = np.asarray(ytr, dtype=float)
    etr_ = exposure(etr, len(ytr))
    pre = Preprocessor().fit(Xtr, categorical)
    Ftr, Fte = _frame(pre, Xtr), _frame(pre, Xte)
    rate = ytr / etr_
    with threadpool(n_threads):
        cv = []
        for params in GRID:
            devs = []
            for tr, va in kfold(len(ytr), INNER_FOLDS, seed):
                m = _model(params, seed).fit(Ftr.iloc[tr], rate[tr], sample_weight=etr_[tr])
                devs.append(mean_deviance(ytr[va], etr_[va] * m.predict(Ftr.iloc[va])))
            cv.append(float(np.mean(devs)))
        best = GRID[int(np.argmin(cv))]
        model = _model(best, seed).fit(Ftr, rate, sample_weight=etr_)
        mu = exposure(ete, len(Fte)) * model.predict(Fte)
        mu_in = etr_ * model.predict(Ftr)
    info = dict(best, n_iter=int(model.n_iter_), cv_deviance=min(cv))
    return dict(mu=mu, kappa=None, kappa_nb=kappa_moment(ytr, mu_in), info=info)
