# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Smoothing-spline ANOVA Poisson regression with main effects (R package gss).

    log mu(x) = log e + b + sum_j f_j(x_j),

each f_j a cubic smoothing spline of a continuous covariate or a nominal effect of a factor,
fitted by gssanova(family = "poisson", offset = log e) with its default random subset of
max(30, 10 n^(2/9)) basis functions. The iteration of gssanova can break down on strongly
overdispersed counts, and its compiled code can then even abort the R process; each fit is
therefore a fresh R process, repeated on failure with another random basis and then with the
smoothing parameters kept at their initial values (skip.iter), as recorded in info.
The linear parts of the splines are unpenalised, and gss requires them to be linearly
independent: continuous covariates that are linear combinations of others on the training fold
(average, maximum and sum of a code metric coincide on files with a single class, for example)
are dropped first. The splines live on the training range of each covariate (extended by 5 per
cent), outside of which gss does not evaluate them; test covariates are therefore clipped to the
training range (constant extrapolation). The method is not run above MAX_TRAIN training rows,
where the n x q x (number of terms) arrays of gss exceed the memory budget of the study.
"""
from __future__ import annotations

import numpy as np

from . import rbridge
from .common import exposure, independent_columns

MAX_TRAIN = 50000


def fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=0, n_threads=1):
    if len(ytr) > MAX_TRAIN:
        return None
    tr, te, spec = rbridge.prepare(Xtr, ytr, etr, Xte, ete, categorical, clip=True)
    cont = spec.loc[spec["kind"] != "factor", "name"].tolist()
    dropped = []
    if cont:
        A = tr[cont].to_numpy(dtype=float)
        A = (A - A.mean(axis=0)) / A.std(axis=0)
        keep = independent_columns(np.hstack([np.ones((len(A), 1)), A]), tol=1e-8)
        keep = set((keep[keep > 0] - 1).tolist())
        dropped = [c for j, c in enumerate(cont) if j not in keep]
        if dropped:
            names = dict(zip(spec["name"], spec["column"]))
            spec = spec[~spec["name"].isin(dropped)].reset_index(drop=True)
            tr, te = tr.drop(columns=dropped), te.drop(columns=dropped)
            dropped = [names[c] for c in dropped]
    base = int(seed) % 100000 + 1
    plan = [(base, 0), (base + 1000, 0), (base, 1), (base + 1000, 1)]
    errors = []
    for attempt, (s, skip) in enumerate(plan, start=1):
        try:
            eta, info = rbridge.run("gss_fit.R", tr, te, spec, [s, skip], n_threads)
            break
        except RuntimeError as exc:
            lines = [ln.strip() for ln in str(exc).splitlines()[1:] if ln.strip()]
            errors.append(" ".join(lines[:2])[:240] if lines else str(exc)[:240])
    else:
        raise RuntimeError("gssanova failed at every attempt: " + " | ".join(errors))
    eta = eta + np.log(exposure(ete, len(eta)))
    mu = np.exp(np.minimum(eta, 60.0))
    return dict(mu=mu, kappa=None, info=dict(nbasis=int(info["nbasis"]), terms=int(info["terms"]),
                                             nlambda=info["nlambda"], attempt=attempt,
                                             skip_iter=bool(skip), clipped_test_inputs=True,
                                             dropped_dependent=dropped, failed_attempts=errors))
