# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Count likelihoods as functions of the natural parameter eta = o + b + f(x).

Poisson:            l(y, eta) = exp(eta) - y eta,
                    l'  = mu - y,  l'' = mu,  mu = exp(eta).
Negative binomial:  l(y, eta) = (y + 1/kappa) log(1 + kappa exp(eta)) - y eta   (kappa > 0 fixed),
                    l'  = (mu - y) / (1 + kappa mu),
                    l'' = mu (1 + kappa y) / (1 + kappa mu)^2.

Both satisfy |l'''| ⩽ l'' (generalised self-concordance with constant one in eta), the
property that controls the Newton iteration. The terms free of eta (log y!, log-Gamma terms)
are added only by the full log-likelihoods used for scoring.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln

ETA_MAX = 60.0  # exp(60) ~ 1e26: beyond any count of interest; guards overflow only


def _mu(eta):
    return np.exp(np.minimum(eta, ETA_MAX))


class Poisson:
    """Canonical Poisson loss; the intercept profiles out in closed form."""

    name = "poisson"
    closed_form_intercept = True

    def value(self, y, eta):
        return float(np.sum(_mu(eta) - y * eta))

    def derivatives(self, y, eta):
        """Return (mu, first derivative, second derivative) of the loss in eta."""
        mu = _mu(eta)
        return mu, mu - y, mu

    def profile_intercept(self, y, base):
        """Minimiser over b of sum_i l(y_i, base_i + b): b = log(sum y / sum exp(base))."""
        sy = float(np.sum(y))
        if sy <= 0.0:
            raise ValueError("all counts are zero: the unpenalised intercept tends to minus infinity")
        shift = float(np.max(base))
        return np.log(sy) - shift - np.log(np.sum(np.exp(base - shift)))

    def loglik(self, y, mu):
        """Full log-likelihood terms log P(Y = y) for each observation."""
        mu = np.maximum(mu, 1e-300)
        return y * np.log(mu) - mu - gammaln(y + 1.0)


class NegativeBinomial:
    """Negative-binomial loss with dispersion kappa (Var = mu + kappa mu^2), kappa fixed."""

    name = "negbin"
    closed_form_intercept = False

    def __init__(self, kappa):
        if kappa <= 0:
            raise ValueError("kappa must be positive; kappa -> 0 is the Poisson loss")
        self.kappa = float(kappa)

    def value(self, y, eta):
        k = self.kappa
        mu = _mu(eta)
        return float(np.sum((y + 1.0 / k) * np.log1p(k * mu) - y * eta))

    def derivatives(self, y, eta):
        k = self.kappa
        mu = _mu(eta)
        d1 = (mu - y) / (1.0 + k * mu)
        d2 = mu * (1.0 + k * y) / (1.0 + k * mu) ** 2
        return mu, d1, d2

    def profile_intercept(self, y, base, b0=None, tol=1e-13, max_iter=100):
        """Minimiser over b of sum_i l(y_i, base_i + b) by safeguarded scalar Newton."""
        if float(np.sum(y)) <= 0.0:
            raise ValueError("all counts are zero: the unpenalised intercept tends to minus infinity")
        if b0 is None:
            b0 = Poisson().profile_intercept(y, base)
        b = float(b0)
        for _ in range(max_iter):
            _, d1, d2 = self.derivatives(y, base + b)
            g, h = float(np.sum(d1)), float(np.sum(d2))
            step = -g / h
            # the scalar loss is convex; damp only to keep exp(eta) finite
            step = float(np.clip(step, -5.0, 5.0))
            b += step
            if abs(step) <= tol * (1.0 + abs(b)):
                break
        return b

    def loglik(self, y, mu):
        k = self.kappa
        r = 1.0 / k
        mu = np.maximum(mu, 1e-300)
        return (gammaln(y + r) - gammaln(r) - gammaln(y + 1.0)
                + r * np.log(r / (r + mu)) + y * np.log(mu / (r + mu)))
