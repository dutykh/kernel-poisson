# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Checks of the identities and algorithms of the article, each against an independent computation.

Run from the root folder of the repository:  python -I -m pytest -q tests
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kernelpoisson import (KernelNewton, NegativeBinomial, Poisson, alo_eta, fit_truncated_newton,
                           gram, leverages, rkhs_pcg)
from kernelpoisson.newton import ProjectedPreconditioner
from kernelpoisson.projected import ProjectedKernelNewton
from kernelpoisson import scores


def problem(N=400, d=3, seed=0, scale=1.0, lengthscale=0.8):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, d))
    e = np.exp(rng.normal(0.0, 0.4, N))
    f0 = np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.5 * X[:, 2] ** 2
    y = rng.poisson(scale * e * np.exp(f0)).astype(float)
    K = gram(X, lengthscale=lengthscale) + 1e-8 * np.eye(N)
    return X, y, np.log(e), K


def test_loss_derivatives_by_finite_differences():
    rng = np.random.default_rng(1)
    y = rng.poisson(3.0, 50).astype(float)
    eta = rng.normal(1.0, 0.5, 50)
    h = 1e-6
    for loss in (Poisson(), NegativeBinomial(0.4)):
        _, d1, d2 = loss.derivatives(y, eta)
        for i in range(5):
            e1 = np.zeros(50); e1[i] = h
            fd1 = (loss.value(y, eta + e1) - loss.value(y, eta - e1)) / (2 * h)
            assert abs(fd1 - d1[i]) < 1e-6
            g_plus = loss.derivatives(y, eta + e1)[1][i]
            g_minus = loss.derivatives(y, eta - e1)[1][i]
            assert abs((g_plus - g_minus) / (2 * h) - d2[i]) < 1e-6
        # generalised self-concordance |l'''| ⩽ l''
        g3 = (loss.derivatives(y, eta + h)[2] - loss.derivatives(y, eta - h)[2]) / (2 * h)
        assert np.all(np.abs(g3) <= d2 * (1 + 1e-5))


def test_preconditioned_newton_matches_exact_newton():
    X, y, o, K = problem()
    lam = 2.0
    exact = KernelNewton(K, y, o).solve(lam, exact=True)
    pcg = KernelNewton(K, y, o).solve(lam)
    assert abs(pcg.objective - exact.objective) <= 1e-9 * abs(exact.objective)
    assert np.max(np.abs(pcg.mu - exact.mu)) <= 1e-8 * np.max(exact.mu)


def test_balance_at_every_iterate_and_at_the_fit():
    X, y, o, K = problem(seed=2)
    kn = KernelNewton(K, y, o)
    fit = kn.solve(1.0, max_newton=2)          # stopped early: still balanced
    assert abs(fit.mu.sum() - y.sum()) <= 1e-10 * y.sum()
    fit = kn.solve(1.0)
    assert abs(fit.mu.sum() - y.sum()) <= 1e-10 * y.sum()


def test_observed_minus_expected_identity():
    X, y, o, K = problem(seed=3)
    lam = 1.5
    fit = KernelNewton(K, y, o).solve(lam, tol=1e-13)
    rng = np.random.default_rng(4)
    x0 = rng.uniform(-1, 1, (20, X.shape[1]))
    k0 = gram(x0, X, lengthscale=0.8)
    lhs = k0 @ (y - fit.mu)
    rhs = lam * (k0 @ fit.alpha)
    assert np.max(np.abs(lhs - rhs)) <= 1e-7 * max(1.0, np.max(np.abs(rhs)))


def test_entropy_dual_stationarity():
    # log(mu/e) + K (mu - y)/lam is constant (= b) at the fit
    X, y, o, K = problem(seed=5)
    lam = 0.7
    fit = KernelNewton(K, y, o).solve(lam, tol=1e-13)
    grad = np.log(fit.mu) - o + K @ (fit.mu - y) / lam
    assert np.max(np.abs(grad - fit.b)) <= 1e-7


def test_linear_kernel_is_the_ridge_poisson_glm():
    from sklearn.linear_model import PoissonRegressor
    rng = np.random.default_rng(6)
    N = 300
    X = rng.normal(size=(N, 3))
    y = rng.poisson(np.exp(0.2 + X @ np.array([0.4, -0.3, 0.2]))).astype(float)
    lam = 5.0
    K = X @ X.T + 1e-10 * np.eye(N)
    fit = KernelNewton(K, y).solve(lam, exact=True, tol=1e-14)
    ridge = PoissonRegressor(alpha=lam / N, tol=1e-12, max_iter=10000).fit(X, y)
    assert np.max(np.abs(fit.mu - ridge.predict(X))) <= 1e-5 * np.max(fit.mu)


def test_truncated_directions_are_descent_directions():
    X, y, o, K = problem(seed=7, scale=20.0)
    lam = 1.0
    kn = KernelNewton(K, y, o)
    b = kn._profile(np.zeros(len(y)))
    _, g, w = Poisson().derivatives(y, o + b)
    r = g.copy()
    L = np.sort(np.random.default_rng(0).choice(len(y), 60, replace=False))
    P = ProjectedPreconditioner(K, w, lam, L)
    for iters in (1, 2, 3, 5):
        d, Kd, k, _ = rkhs_pcg(K, w, lam, r, P, forcing=0.0, max_iter=iters)
        assert float(r @ Kd) < 0.0


def test_leave_one_out_formula_is_second_order():
    X, y, o, K = problem(N=250, seed=8)
    lam = 1.0
    fit = KernelNewton(K, y, o).solve(lam, exact=True, tol=1e-14)
    h, hw = leverages(K, fit.mu, lam)
    eta_alo = alo_eta(fit.eta, fit.mu - y, fit.mu, h, hw)
    errors, shifts = [], []
    for i in range(0, 250, 25):
        keep = np.ones(250, bool); keep[i] = False
        sub = KernelNewton(K[np.ix_(keep, keep)], y[keep], o[keep]).solve(
            lam, alpha0=fit.alpha[keep], exact=True, tol=1e-14)
        eta_loo = o[i] + sub.b + K[i, keep] @ sub.alpha
        errors.append(abs(eta_alo[i] - eta_loo))
        shifts.append(abs(eta_loo - fit.eta[i]))
    assert np.median(errors) < 0.05 * np.median(shifts)


def test_projected_estimator_with_all_landmarks_is_the_full_estimator():
    X, y, o, K = problem(N=200, seed=9)
    lam = 1.0
    full = KernelNewton(gram(X, lengthscale=0.8), y, o).solve(lam, exact=True, tol=1e-13)
    proj = ProjectedKernelNewton(X, y, o, landmarks=np.arange(200), lengthscale=0.8).solve(lam, tol=1e-13)
    assert np.max(np.abs(proj["mu"] - full.mu)) <= 1e-6 * np.max(full.mu)


def test_negative_binomial_reduces_to_poisson_as_kappa_vanishes():
    X, y, o, K = problem(seed=10)
    lam = 1.0
    pois = KernelNewton(K, y, o).solve(lam, exact=True, tol=1e-13)
    nb = KernelNewton(K, y, o, loss=NegativeBinomial(1e-7)).solve(lam, exact=True, tol=1e-13)
    assert np.max(np.abs(nb.mu - pois.mu)) <= 1e-4 * np.max(pois.mu)


def test_truncated_newton_reaches_the_same_optimum_eventually():
    X, y, o, K = problem(N=200, seed=11)
    lam = 2.0
    exact = KernelNewton(K, y, o).solve(lam, exact=True, tol=1e-14)
    tn = fit_truncated_newton(K, y, o, lam=lam, max_outer=400, max_cg=2000, cg_tol=1e-10, tol=1e-14)
    assert abs(tn["objective"] - exact.objective) <= 1e-6 * abs(exact.objective)


def test_scores_against_direct_formulas():
    y = np.array([0, 1, 3, 7], dtype=float)
    mu = np.array([0.5, 1.2, 2.5, 6.0])
    from scipy import stats
    assert np.allclose(scores.log_score(y, mu), -stats.poisson(mu).logpmf(y))
    k = np.arange(0, 200)
    rps_direct = [np.sum((stats.poisson(m).cdf(k) - (yy <= k)) ** 2) for yy, m in zip(y, mu)]
    assert np.allclose(scores.ranked_probability(y, mu), rps_direct)
    # the PIT of a correct model is uniform
    rng = np.random.default_rng(12)
    m = rng.gamma(2.0, 2.0, 200000)
    yy = rng.poisson(m)
    assert np.max(np.abs(scores.pit_histogram(yy, m) - 1.0)) < 0.02


def test_kernels_are_symmetric_positive_semidefinite_with_unit_diagonal():
    rng = np.random.default_rng(13)
    Z = rng.normal(size=(60, 3))
    for family in ("gaussian", "matern52"):
        K = gram(Z, lengthscale=0.9, family=family)
        assert np.allclose(K, K.T, atol=1e-14) and np.allclose(np.diag(K), 1.0, atol=1e-12)
        assert np.linalg.eigvalsh(0.5 * (K + K.T)).min() > -1e-10
    # the closed forms at distance one with length scale one
    origin, one = np.zeros((1, 1)), np.ones((1, 1))
    assert abs(gram(origin, one, 1.0, "gaussian")[0, 0] - np.exp(-0.5)) < 1e-14
    s5 = np.sqrt(5.0)
    assert abs(gram(origin, one, 1.0, "matern52")[0, 0] - (1.0 + s5 + 5.0 / 3.0) * np.exp(-s5)) < 1e-14


def test_the_package_exports_the_estimator():
    from kernelpoisson import KernelPoissonRegressor
    assert KernelPoissonRegressor().family == "poisson"
