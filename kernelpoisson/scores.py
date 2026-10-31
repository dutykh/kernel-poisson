# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Scores of predictive count distributions (Poisson or negative binomial with mean mu).

Mean deviance       2 [y log(y/mu) - (y - mu)]           (strictly consistent for the mean)
Logarithmic score   -log P(Y = y)
Ranked probability  sum_k (F(k) - 1{y ⩽ k})^2
Dawid-Sebastiani    (y - m)^2 / s^2 + 2 log s             (m, s^2 the predictive mean, variance)
Non-randomised PIT  Czado, Gneiting and Held (2009)
Interval score      (u - l) + (2/a)(l - y)_+ + (2/a)(y - u)_+ for the central (1 - a) interval
Gini index          of the concentration curve of observed counts against exposure, ordered by
                    the predicted rate (the ranking power of the predictions)
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy import stats


def _dist(mu, kappa):
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    if kappa is None or kappa == 0:
        return stats.poisson(mu)
    r = 1.0 / kappa
    return stats.nbinom(r, r / (r + mu))


def deviance(y, mu):
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    t = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0) - (y - mu)
    return 2.0 * t


def log_score(y, mu, kappa=None):
    """Logarithmic score -log P(Y = y) of the Poisson law, or of the negative binomial law with
    mean mu and variance mu + kappa mu^2, computed from the closed form of its log-probability
    (which stays finite for tiny means, where p = r/(r + mu) rounds to one)."""
    if not kappa:
        return -_dist(mu, kappa).logpmf(np.asarray(y))
    y = np.asarray(y, dtype=float)
    mu = np.maximum(np.asarray(mu, dtype=float), 1e-300)
    r = 1.0 / kappa
    ll = (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0) - r * np.log1p(mu / r)
          + y * (np.log(mu) - np.log(r + mu)))
    return -ll


def dawid_sebastiani(y, mu, kappa=None):
    mu = np.asarray(mu, dtype=float)
    var = mu if not kappa else mu + kappa * mu * mu
    return (np.asarray(y) - mu) ** 2 / var + np.log(var)


def ranked_probability(y, mu, kappa=None, tail=1e-12):
    """Ranked probability score, the sum truncated where both terms vanish."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    d = _dist(mu, kappa)
    kmax = int(max(np.max(y), np.max(d.ppf(1.0 - tail)))) + 1
    out = np.zeros(len(y))
    # sum over k in blocks to bound memory
    for start in range(0, kmax + 1, 256):
        k = np.arange(start, min(kmax, start + 255) + 1)[None, :]
        F = _dist(mu[:, None], kappa).cdf(k)
        out += np.sum((F - (y[:, None] <= k)) ** 2, axis=1)
    return out


def pit_histogram(y, mu, kappa=None, bins=10):
    """Non-randomised PIT: increments of the mean conditional PIT distribution over the bins."""
    y = np.asarray(y, dtype=float)
    d = _dist(mu, kappa)
    P1 = d.cdf(y)
    P0 = np.where(y > 0, d.cdf(y - 1.0), 0.0)
    u = np.linspace(0.0, 1.0, bins + 1)[None, :]
    width = np.maximum(P1 - P0, 1e-300)[:, None]
    F = np.clip((u - P0[:, None]) / width, 0.0, 1.0)
    Fbar = F.mean(axis=0)
    return np.diff(Fbar) * bins            # equals one on every bin under calibration


def interval(mu, kappa=None, level=0.9):
    d = _dist(mu, kappa)
    a = 1.0 - level
    return d.ppf(a / 2.0), d.ppf(1.0 - a / 2.0)


def interval_score(y, lo, hi, level=0.9):
    a = 1.0 - level
    y = np.asarray(y, dtype=float)
    return (hi - lo) + (2.0 / a) * np.maximum(lo - y, 0.0) + (2.0 / a) * np.maximum(y - hi, 0.0)


def gini(y, mu, exposure=None):
    """Gini index of the ordered concentration curve (higher means better ranking)."""
    y = np.asarray(y, dtype=float)
    e = np.ones(len(y)) if exposure is None else np.asarray(exposure, dtype=float)
    rate = np.asarray(mu, dtype=float) / e
    order = np.argsort(-rate, kind="mergesort")
    ce = np.concatenate([[0.0], np.cumsum(e[order]) / e.sum()])
    cy = np.concatenate([[0.0], np.cumsum(y[order]) / y.sum()])
    area = np.trapezoid(cy, ce)
    return 2.0 * area - 1.0


def rootogram(y, mu, kappa=None, kmax=None):
    """Observed and expected frequencies of each count value."""
    y = np.asarray(y, dtype=int)
    kmax = int(np.max(y)) if kmax is None else int(kmax)
    k = np.arange(kmax + 1)
    observed = np.bincount(np.minimum(y, kmax + 1), minlength=kmax + 2)[: kmax + 1].astype(float)
    expected = _dist(np.asarray(mu)[:, None], kappa).pmf(k[None, :]).sum(axis=0)
    return k, observed, expected


def summary(y, mu, exposure=None, kappa=None, null_mu=None):
    """All scalar scores, averaged per observation, for one set of predictions."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    out = dict(
        deviance=float(np.mean(deviance(y, mu))),
        log_score=float(np.mean(log_score(y, mu, kappa))),
        rps=float(np.mean(ranked_probability(y, mu, kappa))),
        dss=float(np.mean(dawid_sebastiani(y, mu, kappa))),
        rmse=float(np.sqrt(np.mean((y - mu) ** 2))),
        mae=float(np.mean(np.abs(y - mu))),
        balance=float(np.sum(mu) / max(np.sum(y), 1e-300)),
    )
    lo, hi = interval(mu, kappa, 0.9)
    out["coverage90"] = float(np.mean((y >= lo) & (y <= hi)))
    out["interval90"] = float(np.mean(interval_score(y, lo, hi, 0.9)))
    if null_mu is not None:
        out["d2"] = 1.0 - out["deviance"] / float(np.mean(deviance(y, null_mu)))
    if exposure is not None:
        out["gini"] = float(gini(y, mu, exposure))
    return out
