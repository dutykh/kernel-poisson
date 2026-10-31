# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""The kernel Poisson and kernel negative-binomial estimators of the article
(kernelpoisson.estimator.KernelPoissonRegressor), tuned by approximate leave-one-out along the
regularisation path. Above the estimator's full_max training rows (6000) it uses its landmark
(projected) version by itself."""
from __future__ import annotations

import numpy as np

from .common import as_frame, exposure, factor_names, threadpool


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1, family="poisson"):
    from kernelpoisson.estimator import KernelPoissonRegressor

    ytr = np.asarray(ytr, dtype=float)
    Xtr, Xte = as_frame(Xtr), as_frame(Xte)
    idx = tuple(Xtr.columns.get_loc(c) for c in factor_names(Xtr, categorical))
    dtype = object if idx else float
    with threadpool(n_threads):
        est = KernelPoissonRegressor(family=family, categorical=idx, seed=seed)
        est.fit(Xtr.to_numpy(dtype=dtype), ytr, exposure(etr, len(ytr)))
        mu = est.predict(Xte.to_numpy(dtype=dtype), exposure(ete, len(Xte)))
    kappa = float(est.kappa_) if family == "negbin" else None
    # effective number of parameters (sum of the leverages) at the selected point of the path
    df = [q["df"] for q in est.path_ if q["lam"] == est.lam_ and q["lengthscale"] == est.lengthscale_
          and q["kappa"] == est.kappa_]
    info = dict(family=family, lengthscale=float(est.lengthscale_), lam=float(est.lam_),
                alo=float(est.best_["alo"]), mode=est.best_["mode"], path_points=len(est.path_),
                kappa=kappa, df=float(df[0]) if df else None, events=float(np.sum(ytr)),
                landmarks=int(len(est.best_["landmarks"])) if est.best_["mode"] == "projected" else None)
    return dict(mu=mu, kappa=kappa, info=info)
