# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Competitor methods of the benchmark study.

Every method is a function

    fit_predict(Xtr, ytr, etr, Xte, ete, categorical, seed, n_threads) -> dict | None

that fits on the training fold (covariates Xtr, counts ytr, exposures etr or None) and returns
dict(mu=predicted means on the test fold, kappa=dispersion of a negative-binomial law or None,
info=dict of the tuned settings), or None when the method does not apply at that size. Mean-only
methods also return kappa_nb, the moment estimate of a negative-binomial dispersion from the
in-sample means. All tuning uses the training fold only.
"""
METHODS = {
    "glm_poisson": ("baselines.glm", dict(family="poisson")),
    "glm_negbin": ("baselines.glm", dict(family="negbin")),
    "gam_poisson": ("baselines.gam", dict(family="poisson")),
    "gam_negbin": ("baselines.gam", dict(family="negbin")),
    "gss": ("baselines.gss", {}),
    "ktweedie": ("baselines.ktweedie_r", {}),
    "boosting": ("baselines.boosting", {}),
    "forest": ("baselines.forest", {}),
    "network": ("baselines.network", {}),
    "gp_laplace": ("baselines.gp_laplace", {}),
    "krr_log": ("baselines.krr_log", {}),
    "ours_poisson": ("baselines.ours", dict(family="poisson")),
    "ours_negbin": ("baselines.ours", dict(family="negbin")),
}


def get(method):
    """The fit_predict function of a method and its fixed keyword arguments."""
    import importlib
    module, kwargs = METHODS[method]
    return importlib.import_module(module).fit_predict, dict(kwargs)
