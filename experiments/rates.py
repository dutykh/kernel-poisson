#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Rates of kernel Poisson regression in the expected number of events.

Design. X uniform on the circle [0, 1); exposures E = s * V with V log-normal (mean one);
counts Y ~ Poisson(E exp(b0 + f0(X))), or negative binomial with dispersion kappa_d.
Kernel: the periodic Sobolev kernel without constant term,
    k(x, x') = 2 sum_{j=1}^{J} j^{-a} cos(2 pi j (x - x')),
whose eigenvalues j^{-a} (each twice) give the capacity d(lam) of order lam^{-1/a}. Its features
phi_{2j-1} = (2 j^{-a})^{1/2} cos(2 pi j x), phi_{2j} = (2 j^{-a})^{1/2} sin(2 pi j x) make the
full estimator a ridge-penalised Poisson regression in 2J features, solved exactly by the
profiled Newton iteration.

Truths. "smooth": f0 = 0.8 cos(2 pi x) + 0.5 sin(4 pi x) (every source condition holds, predicted
rate Lambda^{-2a/(1+2a)}); "rough": f0 = c sum_j j^{-(a+1)/2 - 0.05} cos(2 pi j x + u_j) (just inside
the reproducing-kernel space, predicted rate Lambda^{-a/(1+a)}).

Error. The excess risk per event, E[mu0 psi(eta_hat - eta0)] / E[mu0] with psi(t) = e^t - 1 - t,
computed exactly on a grid of the circle (the exposure cancels).

Regimes at equal Lambda = N s E[exp(b0 + f0)]: mean count per observation 0.1, 1 and 10.
Tuning: the oracle lam (smallest error on the path) and the approximate leave-one-out lam.
"""
from __future__ import annotations

import argparse
import itertools
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results" / "rates"
J = 300
GRID = np.linspace(0.0, 1.0, 4000, endpoint=False)


def features(x, a):
    j = np.arange(1, J + 1)
    w = np.sqrt(2.0 * j ** (-float(a)))
    arg = 2.0 * np.pi * np.outer(x, j)
    Phi = np.empty((len(x), 2 * J))
    Phi[:, 0::2] = np.cos(arg) * w
    Phi[:, 1::2] = np.sin(arg) * w
    return Phi


def truth(kind, a, rng_seed=0):
    if kind == "smooth":
        return lambda x: 0.8 * np.cos(2 * np.pi * x) + 0.5 * np.sin(4 * np.pi * x)
    rng = np.random.default_rng(rng_seed)
    j = np.arange(1, J + 1)
    c = j ** (-(a + 1.0) / 2.0 - 0.05)
    u = rng.uniform(0, 2 * np.pi, J)
    vals = np.cos(2 * np.pi * np.outer(GRID, j) + u) @ c
    scale = 1.0 / np.max(np.abs(vals))
    return lambda x: scale * (np.cos(2 * np.pi * np.outer(x, j) + u) @ c)


def profiled_newton(Phi, y, o, lam, theta0=None, tol=1e-11, max_iter=60):
    """Exact profiled Newton for sum[exp(o + b + Phi theta) - y (o + b + Phi theta)] + lam/2 |theta|^2."""
    p = Phi.shape[1]
    theta = np.zeros(p) if theta0 is None else theta0.copy()
    f = Phi @ theta
    Y = y.sum()

    def prof(f):
        base = o + f
        sh = base.max()
        return np.log(Y) - sh - np.log(np.sum(np.exp(base - sh)))

    def obj(theta, f, b):
        eta = o + b + f
        return np.sum(np.exp(eta) - y * eta) + 0.5 * lam * theta @ theta

    b = prof(f)
    F = obj(theta, f, b)
    for _ in range(max_iter):
        eta = o + b + f
        mu = np.exp(eta)
        g = Phi.T @ (mu - y) + lam * theta
        PW = Phi * mu[:, None]
        H = Phi.T @ PW
        v = Phi.T @ mu
        H -= np.outer(v, v) / mu.sum()
        H[np.diag_indices_from(H)] += lam
        d = -sla.cho_solve(sla.cho_factor(H, lower=True, check_finite=False), g, check_finite=False)
        Fd = Phi @ d
        slope = g @ d
        t = 1.0
        while True:
            th, ff = theta + t * d, f + t * Fd
            bb = prof(ff)
            Fn = obj(th, ff, bb)
            if Fn <= F + 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        theta, f, b, F = th, ff, bb, Fn
        if -slope <= tol * max(1.0, Y) and t == 1.0:
            break
    return theta, b, f


def alo_deviance(Phi, y, o, theta, b, lam):
    """Approximate leave-one-out mean deviance (intercept included in the smoother)."""
    eta = o + b + Phi @ theta
    mu = np.exp(eta)
    D = np.hstack([np.ones((len(y), 1)), Phi])
    A = D.T @ (D * mu[:, None])
    A[1:, 1:] += lam * np.eye(Phi.shape[1])
    L = sla.cholesky(A, lower=True, check_finite=False)
    V = sla.solve_triangular(L, D.T, lower=True, check_finite=False)
    q = np.einsum("ij,ij->j", V, V)
    h = mu * q
    eta_loo = eta + q * (mu - y) / (1.0 - h)
    mu_loo = np.exp(eta_loo)
    dev = 2.0 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu_loo), 0.0) - (y - mu_loo))
    return float(dev.mean())


def excess_risk(theta, b, b0, f0_grid, Phi_grid):
    delta = (b - b0) + Phi_grid @ theta - f0_grid
    w = np.exp(f0_grid)
    return float(np.sum(w * (np.exp(delta) - 1.0 - delta)) / np.sum(w))


def one_run(kind, a, Lam, mean_count, rep, kappa_d=0.0, b0=0.0):
    seed = zlib.crc32(f"{kind}|{a}|{Lam}|{mean_count}|{rep}|{kappa_d}".encode())
    rng = np.random.default_rng(seed)
    f0 = truth(kind, a)
    f0_grid = f0(GRID)
    Phi_grid = features(GRID, a)
    m0 = float(np.mean(np.exp(b0 + f0_grid)))
    N = max(20, int(round(Lam / mean_count)))
    s = Lam / (N * m0)
    x = rng.uniform(size=N)
    V = rng.lognormal(-0.125, 0.5, N)              # mean one
    e = s * V
    mu0 = e * np.exp(b0 + f0(x))
    if kappa_d > 0:
        y = rng.poisson(rng.gamma(1.0 / kappa_d, kappa_d * mu0)).astype(float)
    else:
        y = rng.poisson(mu0).astype(float)
    if y.sum() == 0:
        return None
    Phi = features(x, a)
    o = np.log(e)
    Yhat = y.sum()
    lam_ev = np.logspace(0, -7, 29)
    best_or = (np.inf, None)
    best_alo = (np.inf, None, None)
    theta = None
    for le in lam_ev:
        lam = le * Yhat
        theta, b, f = profiled_newton(Phi, y, o, lam, theta0=theta)
        risk = excess_risk(theta, b, b0, f0_grid, Phi_grid)
        alo = alo_deviance(Phi, y, o, theta, b, lam)
        if risk < best_or[0]:
            best_or = (risk, le)
        if alo < best_alo[0]:
            best_alo = (alo, le, risk)
    return dict(kind=kind, a=a, Lambda_target=Lam, mean_count=mean_count, kappa_d=kappa_d,
                rep=rep, N=N, Lambda=float(np.sum(mu0)), events=float(Yhat),
                lam_oracle=best_or[1], risk_oracle=best_or[0],
                lam_alo=best_alo[1], risk_alo=best_alo[2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--part", default="all", choices=["all", "poisson", "overdispersed"])
    ap.add_argument("--out", default=str(RESULTS / "rates.csv"))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(map(tuple, old[["kind", "a", "Lambda_target", "mean_count", "kappa_d", "rep"]].values.tolist()))
    jobs = []
    if args.part in ("all", "poisson"):
        for kind, a, Lam, mc, rep in itertools.product(["smooth", "rough"], [2], [300, 1000, 3000, 10000, 30000],
                                                       [0.1, 1.0, 10.0], range(args.reps)):
            jobs.append((kind, a, Lam, mc, rep, 0.0))
    if args.part in ("all", "overdispersed"):
        for kd, Lam, rep in itertools.product([0.5, 2.0], [300, 1000, 3000, 10000, 30000], range(args.reps)):
            jobs.append(("smooth", 2, Lam, 1.0, rep, kd))
    t0 = time.time()
    for k, (kind, a, Lam, mc, rep, kd) in enumerate(jobs):
        key = (kind, a, Lam, mc, kd, rep)
        if key in done:
            continue
        row = one_run(kind, a, Lam, mc, rep, kappa_d=kd)
        if row is None:
            continue
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        if k % 25 == 0:
            print(f"{k + 1}/{len(jobs)} {kind} a={a} Lambda={Lam} mean={mc} kappa_d={kd} rep={rep} "
                  f"N={row['N']} oracle={row['risk_oracle']:.3e} alo={row['risk_alo']:.3e} "
                  f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
