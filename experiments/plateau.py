#!/usr/bin/env python3
# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Excess risk per event at a fixed number of observations, as the exposures grow.

Same model, kernel and rough truth as rates.py (periodic Sobolev kernel of order a = 2, log-rate
just inside the reproducing-kernel space), but the number of observations N is held fixed while
the exposure scale, hence the expected number of events Lambda, grows. The rate theorem predicts
that the risk levels off as the exposures grow: with N observations no estimator does better than a
level that depends on N and not on the exposures. Tuning: the oracle penalty on a grid.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rates import GRID, RESULTS, excess_risk, features, profiled_newton, truth  # noqa: E402


def one_run(N, Lam, rep, kind="rough", a=2, b0=0.0):
    seed = zlib.crc32(f"plateau|{kind}|{a}|{N}|{Lam}|{rep}".encode())
    rng = np.random.default_rng(seed)
    f0 = truth(kind, a)
    f0_grid = f0(GRID)
    Phi_grid = features(GRID, a)
    m0 = float(np.mean(np.exp(b0 + f0_grid)))
    s = Lam / (N * m0)
    x = rng.uniform(size=N)
    V = rng.lognormal(-0.125, 0.5, N)              # mean one
    e = s * V
    mu0 = e * np.exp(b0 + f0(x))
    y = rng.poisson(mu0).astype(float)
    if y.sum() == 0:
        return None
    Phi = features(x, a)
    o = np.log(e)
    Yhat = y.sum()
    best = (np.inf, None)
    theta = None
    for le in np.logspace(0, -10, 41):
        theta, b, f = profiled_newton(Phi, y, o, le * Yhat, theta0=theta)
        risk = excess_risk(theta, b, b0, f0_grid, Phi_grid)
        if risk < best[0]:
            best = (risk, le)
    return dict(kind=kind, a=a, N=N, Lambda_target=Lam, rep=rep, Lambda=float(np.sum(mu0)),
                events=float(Yhat), lam_oracle=best[1], risk_oracle=best[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=15)
    ap.add_argument("--out", default=str(RESULTS / "plateau.csv"))
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = Path(args.out)
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(map(tuple, old[["N", "Lambda_target", "rep"]].values.tolist()))
    jobs = list(itertools.product([10, 30, 100], [100, 300, 1000, 3000, 10000, 30000, 100000, 300000, 1000000],
                                  range(args.reps)))
    t0 = time.time()
    for k, (N, Lam, rep) in enumerate(jobs):
        if (N, Lam, rep) in done:
            continue
        row = one_run(N, Lam, rep)
        if row is None:
            continue
        pd.DataFrame([row]).to_csv(out, mode="a", header=not out.exists(), index=False)
        if k % 15 == 0:
            print(f"{k + 1}/{len(jobs)} N={N} Lambda={Lam} rep={rep} risk={row['risk_oracle']:.3e} "
                  f"t={row['lam_oracle']:.1e} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
