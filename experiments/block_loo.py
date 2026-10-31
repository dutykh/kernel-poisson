# Dr. Denys Dutykh (Mathematics Department, Khalifa University of Science and Technology, Abu Dhabi, UAE)
"""Accuracy of the one-step leave-one-block-out predictor against exact refits.

On the simulated data of the leave-one-out experiment (loo_accuracy.py), the observations are
grouped into blocks of neighbours, the cells of a regular grid of the square of the covariates.
At every penalty of a path the script refits the estimator without each block in turn and
records (i) the error of the one-step block predictor together with the bound of the block
theorem, computed from the full fit alone, and (ii) the exact and one-step leave-one-block-out
mean deviances, next to the exact leave-one-out mean deviance.
Outputs: results/loo/blocks_path.csv and results/loo/blocks_points.csv.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from kernelpoisson import KernelNewton, gram  # noqa: E402
from kernelpoisson.losses import Poisson  # noqa: E402
from loo_accuracy import simulate, unit_deviance  # noqa: E402

RESULTS = HERE.parent / "results" / "loo"


def pairing_matrix(K, w, lam):
    """Q_ik = <phi_i, H^{-1} phi_k> = G + lam v v' / (w'v), with G = (K W + lam I)^{-1} K and
    v = (K W + lam I)^{-1} 1 (the leverage lemma of the supplementary material)."""
    n = len(w)
    lu = sla.lu_factor(K * w[None, :] + lam * np.eye(n), check_finite=False)
    G = sla.lu_solve(lu, K, check_finite=False)
    G = 0.5 * (G + G.T)
    v = sla.lu_solve(lu, np.ones(n), check_finite=False)
    return G + lam * np.outer(v, v) / float(w @ v)


def grid_blocks(x, cells):
    """The nonempty cells of a cells x cells grid of the square [-1, 1]^2."""
    ix = np.minimum(((x[:, 0] + 1.0) / 2.0 * cells).astype(int), cells - 1)
    iy = np.minimum(((x[:, 1] + 1.0) / 2.0 * cells).astype(int), cells - 1)
    lab = ix * cells + iy
    return [np.flatnonzero(lab == c) for c in np.unique(lab)]


def one_penalty(K, y, off, lam, blocks, alpha0=None):
    N = len(y)
    kn = KernelNewton(K, y, offset=off, loss=Poisson(), preconditioner="none")
    fit = kn.solve(lam, alpha0=alpha0, tol=1e-13, exact=True)
    eta = off + fit.b + K @ fit.alpha
    mu = np.exp(eta)
    g, w = mu - y, mu
    Q = pairing_matrix(K, w, lam)
    q = np.diag(Q)
    eta_t, eta_ex = np.empty(N), np.empty(N)
    bound, cond, block_id = np.empty(N), np.empty(N), np.empty(N, dtype=int)
    for k, B in enumerate(blocks):
        QB, wB, gB = Q[np.ix_(B, B)], w[B], g[B]
        SB = np.sqrt(wB)[:, None] * QB * np.sqrt(wB)[None, :]
        sB = float(np.linalg.eigvalsh(SB).max())
        QBp = QB @ np.linalg.inv(np.eye(len(B)) - wB[:, None] * QB)    # Q_B (I - W_B Q_B)^{-1}
        eta_t[B] = eta[B] + QBp @ gB
        cB = float(gB @ QBp @ gB)
        keep = np.setdiff1d(np.arange(N), B)
        AB = np.sqrt(np.max(q[keep]) / (1.0 - sB))                    # part (d) of the theorem
        bound[B] = 1.04 * AB * np.sqrt(np.diag(QBp)) * cB
        cond[B] = AB * np.sqrt(cB)
        block_id[B] = k
        knb = KernelNewton(K[np.ix_(keep, keep)], y[keep], offset=off[keep], loss=Poisson(),
                           preconditioner="none")
        fb = knb.solve(lam, alpha0=fit.alpha[keep], tol=1e-13, exact=True)
        eta_ex[B] = off[B] + fb.b + K[np.ix_(B, keep)] @ fb.alpha
    return fit, eta, eta_t, eta_ex, bound, cond, block_id


def exact_loo(K, y, off, lam, alpha):
    N = len(y)
    out = np.empty(N)
    for i in range(N):
        keep = np.arange(N) != i
        kni = KernelNewton(K[np.ix_(keep, keep)], y[keep], offset=off[keep], loss=Poisson(),
                           preconditioner="none")
        fi = kni.solve(lam, alpha0=alpha[keep], tol=1e-13, exact=True)
        out[i] = off[i] + fi.b + K[i, keep] @ fi.alpha
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=400)
    ap.add_argument("--mean", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cells", type=int, default=5)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    x, e, y = simulate(args.N, args.mean, args.seed)
    off = np.log(e)
    K = gram(x, x, lengthscale=0.5, family="gaussian")
    blocks = grid_blocks(x, args.cells)
    sizes = np.array([len(B) for B in blocks])
    print(f"{len(blocks)} blocks of {sizes.min()} to {sizes.max()} observations", flush=True)
    ts = np.logspace(-1, -6, 16)
    rows, best, alpha0 = [], None, None
    for t in ts:
        lam = t * y.sum()
        fit, eta, eta_t, eta_ex, bound, cond, block_id = one_penalty(K, y, off, lam, blocks, alpha0)
        alpha0 = fit.alpha
        loo = exact_loo(K, y, off, lam, fit.alpha)
        ok = cond <= 0.25
        err = np.abs(eta_t - eta_ex)
        row = dict(t=t, lam=lam, cells=args.cells, blocks=len(blocks),
                   onestep_block_deviance=float(np.mean(unit_deviance(y, eta_t))),
                   exact_block_deviance=float(np.mean(unit_deviance(y, eta_ex))),
                   exact_loo_deviance=float(np.mean(unit_deviance(y, loo))),
                   train_deviance=float(np.mean(unit_deviance(y, eta))),
                   share_condition=float(np.mean(ok)),
                   max_error=float(np.max(err)),
                   max_ratio_error_bound=float(np.max(err[ok] / bound[ok])) if ok.any() else np.nan)
        rows.append(row)
        print(f"t={t:.1e} block one-step={row['onestep_block_deviance']:.5f} "
              f"exact={row['exact_block_deviance']:.5f} loo={row['exact_loo_deviance']:.5f} "
              f"hypothesis={ok.mean():.0%} max err/bound={row['max_ratio_error_bound']:.3f}", flush=True)
        if best is None or row["onestep_block_deviance"] < best[0]:
            best = (row["onestep_block_deviance"], t, eta, eta_t, eta_ex, bound, cond, block_id)
    pd.DataFrame(rows).to_csv(RESULTS / "blocks_path.csv", index=False)
    _, t, eta, eta_t, eta_ex, bound, cond, block_id = best
    pd.DataFrame(dict(t=t, block=block_id, y=y, eta=eta, eta_onestep=eta_t, eta_exact=eta_ex,
                      bound=bound, condition=cond, error=np.abs(eta_t - eta_ex))).to_csv(
        RESULTS / "blocks_points.csv", index=False)
    ok = cond <= 0.25
    worst = f"{np.max((np.abs(eta_t - eta_ex) / bound)[ok]):.3f}" if ok.any() else "none"
    print(f"selected t={t:.1e}; hypothesis holds for {ok.mean():.0%} of the observations; "
          f"max error/bound where it holds = {worst}")


if __name__ == "__main__":
    main()
