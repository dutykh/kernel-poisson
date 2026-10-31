# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Accuracy of the one-step leave-one-out predictor against exact refits.

For simulated Poisson data with exposures and a Gaussian kernel, the script refits the estimator
without each observation in turn, at every penalty of a path, and records (i) the error of the
one-step predictor together with the bound of the leave-one-out theorem, computed from the full fit
alone, and (ii) the exact and approximate leave-one-out mean deviances along the path.
Outputs: results/loo/points.csv and results/loo/path.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from kernelpoisson import KernelNewton, gram  # noqa: E402
from kernelpoisson.alo import alo_eta, leverages  # noqa: E402
from kernelpoisson.losses import Poisson  # noqa: E402

RESULTS = HERE.parent / "results" / "loo"


def unit_deviance(y, eta):
    mu = np.exp(eta)
    with np.errstate(divide="ignore", invalid="ignore"):
        ylogy = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0)), 0.0)
    return 2.0 * (ylogy - y * eta + mu - y)


def simulate(N, mean, seed):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=(N, 2))
    f0 = np.sin(2.0 * x[:, 0]) + 0.5 * x[:, 1] ** 2 - 0.3 * x[:, 0] * x[:, 1]
    e = rng.uniform(0.5, 1.5, N)
    mu0 = mean * e * np.exp(f0 - f0.mean())
    y = rng.poisson(mu0).astype(float)
    return x, e, y


def one_penalty(K, y, off, lam, alpha0=None):
    kn = KernelNewton(K, y, offset=off, loss=Poisson(), preconditioner="none")
    fit = kn.solve(lam, alpha0=alpha0, tol=1e-13, exact=True)
    eta = off + fit.b + K @ fit.alpha
    mu = np.exp(eta)
    g, w = mu - y, mu
    h, hw = leverages(K, w, lam)
    eta_t = alo_eta(eta, g, w, h, hw)
    N = len(y)
    eta_ex = np.empty(N)
    for i in range(N):
        keep = np.arange(N) != i
        kni = KernelNewton(K[np.ix_(keep, keep)], y[keep], offset=off[keep], loss=Poisson(),
                           preconditioner="none")
        fi = kni.solve(lam, alpha0=fit.alpha[keep], tol=1e-13, exact=True)
        eta_ex[i] = off[i] + fi.b + K[i, keep] @ fi.alpha
    q = hw
    c = g * g / w * h / (1.0 - h)
    qmax_others = np.array([np.max(np.delete(q, i)) for i in range(N)])
    A = np.sqrt(qmax_others / (1.0 - h))
    bound = 1.04 * A * np.sqrt(q / (1.0 - h)) * c
    cond = A * np.sqrt(c)
    return fit, eta, eta_t, eta_ex, h, c, bound, cond


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--mean", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    x, e, y = simulate(args.N, args.mean, args.seed)
    off = np.log(e)
    K = gram(x, x, lengthscale=0.5, family="gaussian")
    ts = np.logspace(-1, -6, 16)
    rows, best = [], None
    alpha0 = None
    for t in ts:
        lam = t * y.sum()
        fit, eta, eta_t, eta_ex, h, c, bound, cond = one_penalty(K, y, off, lam, alpha0)
        alpha0 = fit.alpha
        alo_dev = float(np.mean(unit_deviance(y, eta_t)))
        loo_dev = float(np.mean(unit_deviance(y, eta_ex)))
        rows.append(dict(t=t, lam=lam, alo_deviance=alo_dev, loo_deviance=loo_dev,
                         train_deviance=float(np.mean(unit_deviance(y, eta))),
                         dof=float(np.sum(h)), max_error=float(np.max(np.abs(eta_t - eta_ex))),
                         share_condition=float(np.mean(cond <= 0.25))))
        print(f"t={t:.1e} alo={alo_dev:.5f} loo={loo_dev:.5f} dof={np.sum(h):.1f} "
              f"max|err|={np.max(np.abs(eta_t - eta_ex)):.2e}", flush=True)
        if best is None or alo_dev < best[0]:
            best = (alo_dev, t, eta, eta_t, eta_ex, h, c, bound, cond)
    pd.DataFrame(rows).to_csv(RESULTS / "path.csv", index=False)
    _, t, eta, eta_t, eta_ex, h, c, bound, cond = best
    pd.DataFrame(dict(t=t, y=y, eta=eta, eta_alo=eta_t, eta_loo=eta_ex, leverage=h, cook=c,
                      bound=bound, condition=cond,
                      error=np.abs(eta_t - eta_ex), shift=np.abs(eta_ex - eta))).to_csv(
        RESULTS / "points.csv", index=False)
    ok = cond <= 0.25
    worst = f"{np.max((np.abs(eta_t - eta_ex) / bound)[ok]):.3f}" if ok.any() else "none"
    print(f"selected t={t:.1e}; condition holds for {ok.mean():.0%}; "
          f"max error/bound where it holds = {worst}")


if __name__ == "__main__":
    main()
