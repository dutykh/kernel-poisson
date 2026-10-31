#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""The Newton systems of kernel Poisson regression: spectra, preconditioning and cost.

Part A (spectra against the count scale). For data with mean counts scaled by s, at the fit:
the condition number of the coefficient system K M K + lam K of the 2011 method, of the same
operator in the geometry of the reproducing-kernel space (B = K^{1/2} M K^{1/2} + lam I), and of
B preconditioned by the projected Hessian on m landmarks (uniform or Fisher-leverage), with the
bounds 1 + mu_max sigma_1(K)/lam and (1 + sqrt(eps) + eps)/(1 - sqrt(eps)), eps = |M~ - M~_L|/lam.

Part B (cost against N). For exact Newton, the 2011 method, conjugate gradients in the kernel
geometry without and with preconditioning, and L-BFGS, each run until its own stopping rule
(tolerance 1e-12 or its iteration limit): the kernel matrix-vector products (counted in units of
one N x N product), the time, and the relative gap of the objective to a reference solution.

Part C (convergence). Relative objective gap against the elapsed time for one problem.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kernelpoisson import KernelNewton, gram, fit_truncated_newton, leverages, alo_eta  # noqa: E402
from kernelpoisson.landmarks import leverage_upper  # noqa: E402
from kernelpoisson.newton import ProjectedPreconditioner  # noqa: E402

RESULTS = HERE.parent / "results" / "newton"


class CountingKernel(np.ndarray):
    """Gram matrix that counts its products with vectors, in units of one N x N product."""
    count = 0.0
    N = 1

    def __matmul__(self, other):
        other = np.asarray(other)
        ncol = 1 if other.ndim == 1 else other.shape[1]
        CountingKernel.count += self.shape[0] * self.shape[1] * ncol / CountingKernel.N ** 2
        return np.asarray(self) @ other

    def __rmatmul__(self, other):
        other = np.asarray(other)
        nrow = 1 if other.ndim == 1 else other.shape[0]
        CountingKernel.count += self.shape[0] * self.shape[1] * nrow / CountingKernel.N ** 2
        return other @ np.asarray(self)


def data(N, scale, seed, d=3, lengthscale=0.8):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (N, d))
    e = np.exp(rng.normal(0.0, 0.4, N))
    f0 = np.sin(2 * X[:, 0]) * np.cos(X[:, 1]) + 0.5 * X[:, 2] ** 2
    y = rng.poisson(scale * e * np.exp(f0)).astype(float)
    K = gram(X, lengthscale=lengthscale)
    K[np.diag_indices_from(K)] += 1e-8
    return X, y, np.log(e), K


def alo_lambda(K, y, o, grid):
    """lam on a relative grid chosen by approximate leave-one-out deviance."""
    kn = KernelNewton(K, y, o)
    best, alpha = (np.inf, None, None), None
    for lr in grid:
        lam = lr * y.sum()
        fit = kn.solve(lam, alpha0=alpha, tol=1e-9)
        alpha = fit.alpha
        h, hw = leverages(K, fit.mu, lam)
        eta = alo_eta(fit.eta, fit.mu - y, fit.mu, h, hw)
        mu = np.exp(eta)
        dev = np.mean(2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu)))
        if dev < best[0]:
            best = (dev, lam, fit)
    return best[1], best[2]


def generalized_condition(B, P):
    L = np.linalg.cholesky(P)
    Z = sla.solve_triangular(L, B, lower=True)
    Z = sla.solve_triangular(L, Z.T, lower=True)
    ev = np.linalg.eigvalsh(0.5 * (Z + Z.T))
    return ev[-1] / ev[0]


def part_a(reps=3, N=1500):
    rows = []
    rng = np.random.default_rng(1)
    for scale in [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]:
        for rep in range(reps):
            X, y, o, K = data(N, scale, seed=100 * rep + 7)
            lam, fit = alo_lambda(K, y, o, np.logspace(-1, -6, 11))
            mu = fit.mu
            Y = mu.sum()
            M = np.diag(mu) - np.outer(mu, mu) / Y
            evK, U = np.linalg.eigh(K)
            Kh = (U * np.sqrt(np.maximum(evK, 0))) @ U.T
            H_old = K @ M @ K + lam * K
            ev_old = np.linalg.eigvalsh(0.5 * (H_old + H_old.T))
            B = Kh @ M @ Kh + lam * np.eye(N)
            ev_B = np.linalg.eigvalsh(B)
            Mw = np.sqrt(mu)[:, None] * K * np.sqrt(mu)[None, :]
            lev = np.diag(Mw @ np.linalg.inv(Mw + lam * np.eye(N)))
            dexact = lev.sum()
            row = dict(scale=scale, rep=rep, N=N, lam=lam, mu_max=mu.max(), d=dexact,
                       kappa_truncated_newton=ev_old[-1] / max(ev_old[0], 1e-300), kappa_rkhs=ev_B[-1] / ev_B[0],
                       bound_rkhs=1 + mu.max() * evK[-1] / lam)
            for c in (1, 2, 4):
                m = int(min(N, max(10, round(c * dexact))))
                for scheme in ("uniform", "fisher"):
                    p = np.ones(N) / N if scheme == "uniform" else lev / lev.sum()
                    S = np.sort(rng.choice(N, size=m, replace=False, p=p))
                    E = Kh[:, S]
                    Pi = E @ np.linalg.solve(E.T @ E + 1e-12 * np.eye(m), E.T)
                    P = Pi @ (Kh @ M @ Kh) @ Pi + lam * np.eye(N)
                    C = Mw[:, S]
                    Mh = C @ np.linalg.solve(Mw[np.ix_(S, S)] + 1e-12 * np.eye(m), C.T)
                    eps = np.linalg.eigvalsh(Mw - Mh)[-1] / lam
                    row[f"kappa_{scheme}_{c}"] = generalized_condition(B, P)
                    row[f"eps_{scheme}_{c}"] = eps
                    row[f"m_{c}"] = m
            rows.append(row)
            print(f"A scale={scale} rep={rep} mu_max={mu.max():.1f} d={dexact:.1f} "
                  f"kappa truncated={row['kappa_truncated_newton']:.1e} rkhs={row['kappa_rkhs']:.1e} "
                  f"fisher(2d)={row['kappa_fisher_2']:.2f} uniform(2d)={row['kappa_uniform_2']:.2f}", flush=True)
    return pd.DataFrame(rows)


def solve_counted(method, K, y, o, lam, ref_obj, tol_rel=1e-9):
    CountingKernel.N = K.shape[0]
    CK = K.view(CountingKernel)
    CountingKernel.count = 0.0
    t0 = time.perf_counter()
    trace = []
    if method == "exact":
        fit = KernelNewton(CK, y, o).solve(lam, exact=True, tol=1e-12)
        obj = fit.objective
        cg = 0
    elif method == "truncated_newton":
        out = fit_truncated_newton(CK, y, o, lam=lam, max_outer=500, max_cg=200, tol=1e-12)
        obj = out["objective"]
        cg = sum(r["cg"] for r in out["records"])
        trace = [r["objective"] for r in out["records"]]
    elif method in ("rkhs", "uniform", "fisher"):
        pre = {"rkhs": "none", "uniform": "uniform", "fisher": "fisher"}[method]
        kn = KernelNewton(CK, y, o, preconditioner=pre, max_cg=2000, landmark_factor=3.0)
        fit = kn.solve(lam, tol=1e-12)
        obj = fit.objective
        cg = sum(r.cg_iterations for r in fit.records)
        trace = [r.objective for r in fit.records]
    elif method == "lbfgs":
        Y = y.sum()

        def fun(alpha):
            f = CK @ alpha
            base = o + f
            sh = base.max()
            b = np.log(Y) - sh - np.log(np.sum(np.exp(base - sh)))
            mu = np.exp(base + b)
            val = np.sum(mu - y * (base + b)) + 0.5 * lam * alpha @ f
            grad = CK @ (mu - y + lam * alpha)
            trace.append(val)
            return val, grad
        res = minimize(fun, np.zeros(len(y)), jac=True, method="L-BFGS-B",
                       options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-12, maxcor=20))
        obj = res.fun
        cg = res.nit
    seconds = time.perf_counter() - t0
    err = abs(obj - ref_obj) / abs(ref_obj)
    return dict(method=method, matvecs=CountingKernel.count, seconds=seconds, rel_error=err,
                inner=cg), trace


def part_b(sizes=(500, 1000, 2000, 4000, 8000, 16000), scales=(1.0, 30.0), lam_rel=1e-4):
    rows = []
    for N, scale in [(n, s) for s in scales for n in sizes]:
        X, y, o, K = data(N, scale, seed=N)
        lam = lam_rel * y.sum()
        ref = KernelNewton(K, y, o, landmark_factor=4.0).solve(lam, tol=1e-14).objective
        for method in ("exact", "truncated_newton", "rkhs", "uniform", "fisher", "lbfgs"):
            if method == "exact" and N > 4000:
                continue
            if method in ("truncated_newton", "lbfgs", "rkhs") and N > 8000:
                continue
            r, _ = solve_counted(method, K, y, o, lam, ref)
            r.update(N=N, scale=scale, lam=lam)
            rows.append(r)
            print(f"B N={N} {method:8s} matvecs={r['matvecs']:9.1f} time={r['seconds']:7.2f}s "
                  f"rel_err={r['rel_error']:.1e} inner={r['inner']}", flush=True)
        del K
    return pd.DataFrame(rows)


def part_c(N=4000, scale=3.0, lam_rel=1e-4):
    """Convergence histories: relative objective gap against elapsed time, one row per outer step
    (per function evaluation for the quasi-Newton method)."""
    X, y, o, K = data(N, scale, seed=N)
    lam = lam_rel * y.sum()
    ref = KernelNewton(K, y, o, landmark_factor=4.0).solve(lam, tol=1e-14).objective
    rows = []
    for method in ("truncated_newton", "rkhs", "uniform", "fisher"):
        t0 = time.perf_counter()
        if method == "truncated_newton":
            out = fit_truncated_newton(K, y, o, lam=lam, max_outer=500, max_cg=200, tol=1e-12)
            recs = [(r["seconds"], r["objective"]) for r in out["records"]]
        else:
            pre = {"rkhs": "none", "uniform": "uniform", "fisher": "fisher"}[method]
            fit = KernelNewton(K, y, o, preconditioner=pre, max_cg=2000, landmark_factor=3.0).solve(lam, tol=1e-12)
            recs = [(r.seconds, r.objective) for r in fit.records]
        elapsed = 0.0
        for k, (sec, obj) in enumerate(recs):
            elapsed += sec
            rows.append(dict(method=method, step=k, seconds=elapsed, gap=abs(obj - ref) / abs(ref)))
        print(f"C {method}: {len(recs)} steps, {time.perf_counter() - t0:.2f}s, final gap {rows[-1]['gap']:.1e}", flush=True)
    Y = y.sum()
    stamps = []
    t0 = time.perf_counter()

    def fun(alpha):
        f = K @ alpha
        base = o + f
        sh = base.max()
        b = np.log(Y) - sh - np.log(np.sum(np.exp(base - sh)))
        mu = np.exp(base + b)
        val = np.sum(mu - y * (base + b)) + 0.5 * lam * alpha @ f
        grad = K @ (mu - y + lam * alpha)
        stamps.append((time.perf_counter() - t0, val))
        return val, grad
    minimize(fun, np.zeros(len(y)), jac=True, method="L-BFGS-B",
             options=dict(maxiter=20000, maxfun=40000, ftol=1e-16, gtol=1e-12, maxcor=20))
    best = np.inf
    for k, (sec, val) in enumerate(stamps):
        best = min(best, val)
        rows.append(dict(method="lbfgs", step=k, seconds=sec, gap=abs(best - ref) / abs(ref)))
    print(f"C lbfgs: {len(stamps)} evaluations, final gap {rows[-1]['gap']:.1e}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", "a", "b", "c"])
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if args.part in ("all", "a"):
        part_a().to_csv(RESULTS / "spectra.csv", index=False)
    if args.part in ("all", "b"):
        part_b().to_csv(RESULTS / "cost.csv", index=False)
    if args.part in ("all", "c"):
        part_c().to_csv(RESULTS / "trace.csv", index=False)


if __name__ == "__main__":
    main()
