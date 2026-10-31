# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Simulation study with a known truth: where does kernel Poisson regression beat the standard
models of count regression?

Covariates are uniform on [0, 1]^p, exposures log-normal with mean one, and counts Poisson with
mean e exp(b + f0(x)), the intercept b setting the mean count per observation (0.3 or 3). Four
log-rates are used: a smooth interaction, an additive function, a threshold function and a linear
function. Each method is fitted on N = 1000 observations and judged by the excess risk per event
against the truth, E[mu0 psi(eta_hat - eta0)] / E[mu0] with psi(t) = e^t - 1 - t, estimated on
20 000 fresh covariate values (the exposure cancels). Writes results/simulation/NAME.csv.

Usage, from the root folder of the repository:
    python -I experiments/simulation.py --truth NAME --mean M [--reps R]
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import zlib
from pathlib import Path

_T = os.environ.get("SIM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = _T

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import baselines  # noqa: E402

RESULTS = HERE.parent / "results" / "simulation"
METHODS = ("glm_poisson", "gam_poisson", "boosting", "forest", "ours_poisson")
N_TRAIN, N_TEST = 1000, 20000


def truth(name):
    if name == "interaction":
        return 3, lambda x: np.sin(2 * np.pi * x[:, 0]) * np.cos(np.pi * x[:, 1]) + 0.5 * x[:, 2]
    if name == "additive":
        return 4, lambda x: (np.sin(2 * np.pi * x[:, 0]) + 0.8 * (x[:, 1] - 0.5) ** 2 * 4
                             + 0.5 * np.exp(-8 * (x[:, 2] - 0.3) ** 2) - 0.4 * x[:, 3])
    if name == "threshold":
        return 3, lambda x: (0.8 * (x[:, 0] > 0.5) + 0.6 * ((x[:, 1] > 0.3) & (x[:, 2] > 0.6))
                             - 0.4 * (x[:, 2] < 0.2))
    if name == "linear":
        return 4, lambda x: x @ np.array([0.8, -0.6, 0.4, 0.2])
    raise ValueError(name)


def excess_risk_per_event(eta_hat, eta0, mu0):
    d = eta_hat - eta0
    return float(np.sum(mu0 * (np.expm1(d) - d)) / np.sum(mu0))


def one_rep(name, mean, rep):
    p, f0 = truth(name)
    rng = np.random.default_rng(zlib.crc32(f"{name}|{mean}|{rep}".encode()))
    xtr = rng.uniform(size=(N_TRAIN, p))
    xte = rng.uniform(size=(N_TEST, p))
    m0 = float(np.mean(np.exp(f0(rng.uniform(size=(200000, p))))))
    b = np.log(mean / m0)
    etr = rng.lognormal(-0.125, 0.5, N_TRAIN)
    mu_tr = etr * np.exp(b + f0(xtr))
    ytr = rng.poisson(mu_tr).astype(float)
    eta0_te = b + f0(xte)
    mu0_te = np.exp(eta0_te)
    ete = np.ones(N_TEST)
    cols = [f"x{j + 1}" for j in range(p)]
    Xtr = pd.DataFrame(xtr, columns=cols)
    Xte = pd.DataFrame(xte, columns=cols)
    rows = []
    for method in METHODS:
        fit_predict, kw = baselines.get(method)
        t0 = time.perf_counter()
        try:
            out = fit_predict(Xtr, ytr, etr, Xte, ete, categorical=(), seed=rep, n_threads=int(_T), **kw)
            status, msg = ("ok", "") if out is not None else ("skipped", "")
        except Exception as exc:          # recorded, never hidden
            out, status, msg = None, "failed", repr(exc)[:300]
        sec = time.perf_counter() - t0
        risk = np.nan
        if out is not None:
            eta_hat = np.log(np.maximum(out["mu"], 1e-300)) - np.log(ete)
            risk = excess_risk_per_event(eta_hat, eta0_te, mu0_te)
        rows.append(dict(truth=name, mean_count=mean, rep=rep, method=method, status=status,
                         risk=risk, seconds=sec, events=float(ytr.sum()), message=msg))
        print(f"{name} mean={mean} rep={rep} {method:>13s} risk={risk:.4g} [{sec:.1f}s] {msg}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True, choices=["interaction", "additive", "threshold", "linear"])
    ap.add_argument("--mean", type=float, required=True)
    ap.add_argument("--reps", type=int, default=20)
    a = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{a.truth}_{a.mean:g}.csv"
    done = set()
    if out.exists():
        old = pd.read_csv(out)
        done = set(old["rep"].unique())
    for rep in range(a.reps):
        if rep in done:
            continue
        rows = one_rep(a.truth, a.mean, rep)
        pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
